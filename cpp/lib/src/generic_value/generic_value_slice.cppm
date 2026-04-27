module;

export module hasty_generic_value_mod:slice;

import std;
import hasty_util_mod;
import hasty_tensor_mod;

namespace hasty {
namespace slice {

struct Token {
    enum class Kind { LBracket, RBracket, String, Integer, Colon, Comma, Ellipsis, End };
    Kind kind;
    std::string text; // for String / Integer tokens
};

struct Lexer {
    const std::string& src;
    std::size_t pos = 0;

    char peek() const { return pos < src.size() ? src[pos] : '\0'; }
    char get()        { return pos < src.size() ? src[pos++] : '\0'; }
    void skip_ws()    { while (pos < src.size() && src[pos] == ' ') ++pos; }

    Token next() {
        skip_ws();
        if (pos >= src.size()) return {Token::Kind::End, {}};
        char c = peek();
        if (c == '[')  { ++pos; return {Token::Kind::LBracket,  {}}; }
        if (c == ']')  { ++pos; return {Token::Kind::RBracket,  {}}; }
        if (c == ':')  { ++pos; return {Token::Kind::Colon,     {}}; }
        if (c == ',')  { ++pos; return {Token::Kind::Comma,     {}}; }
        if (c == '"' || c == '\'') {
            char q = get();
            std::string s;
            while (pos < src.size() && src[pos] != q) s += src[pos++];
            if (pos < src.size()) ++pos; // consume closing quote
            return {Token::Kind::String, std::move(s)};
        }
        if (c == '.' && src.size() - pos >= 3 &&
            src[pos+1] == '.' && src[pos+2] == '.') {
            pos += 3;
            return {Token::Kind::Ellipsis, {}};
        }
        // integer (possibly negative)
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            std::string s;
            if (c == '-') { s += get(); }
            while (pos < src.size() && std::isdigit(static_cast<unsigned char>(src[pos])))
                s += src[pos++];
            return {Token::Kind::Integer, std::move(s)};
        }
        throw std::runtime_error(
            std::string("gv_slice: unexpected char '") + c + "' at offset " + std::to_string(pos));
    }

    Token peek_token() {
        auto saved = pos;
        auto tok   = next();
        pos = saved;
        return tok;
    }
};

std::optional<i64> parse_opt_int(Lexer& lex) {
    auto tok = lex.peek_token();
    if (tok.kind == Token::Kind::Integer) {
        lex.next();
        return std::stoll(tok.text);
    }
    return std::nullopt;
}

TensorIndex parse_dim_spec(Lexer& lex) {
    auto tok = lex.peek_token();

    // Ellipsis
    if (tok.kind == Token::Kind::Ellipsis) {
        lex.next();
        return TensorIndex(Ellipsis);
    }

    // Attempt to parse optional start integer, then check for colon
    std::optional<i64> start = std::nullopt;
    bool saw_int = false;
    if (tok.kind == Token::Kind::Integer) {
        lex.next();
        start = std::stoll(tok.text);
        saw_int = true;
    }

    // Peek next: if it's colon → slice; if it's comma/] → plain integer index
    auto nxt = lex.peek_token();
    if (nxt.kind != Token::Kind::Colon) {
        // Plain integer index (no colon follows)
        if (saw_int) return TensorIndex(*start);
        // Bare ':' with no leading integer is handled below; ':' leads to slice
        throw std::runtime_error("gv_slice: expected integer or slice");
    }

    // It's a slice: consume the colon
    lex.next();
    std::optional<i64> stop = parse_opt_int(lex);
    std::optional<i64> step = std::nullopt;
    if (lex.peek_token().kind == Token::Kind::Colon) {
        lex.next();
        step = parse_opt_int(lex);
    }
    return TensorIndex(Slice(start, stop, step));
}

// Parse the content inside [...] and return the list of TensorIndex-es
// for a tensor multi-index.  The opening '[' has already been consumed.
std::vector<TensorIndex> parse_tensor_indices(Lexer& lex) {
    std::vector<TensorIndex> result;
    // Special case: first token might be ':' (full slice) — handle before parse_dim_spec
    // which expects an integer-or-ellipsis first.
    // We handle ':' as a prefix here so parse_dim_spec doesn't need to.
    auto handle_colon_prefix = [&]() -> bool {
        if (lex.peek_token().kind == Token::Kind::Colon) {
            lex.next();
            // check for step: ::N
            if (lex.peek_token().kind == Token::Kind::Colon) {
                lex.next();
                auto step = parse_opt_int(lex);
                result.push_back(TensorIndex(Slice(std::nullopt, std::nullopt, step)));
            } else {
                auto stop = parse_opt_int(lex);
                if (lex.peek_token().kind == Token::Kind::Colon) {
                    lex.next();
                    auto step = parse_opt_int(lex);
                    result.push_back(TensorIndex(Slice(std::nullopt, stop, step)));
                } else {
                    result.push_back(TensorIndex(Slice(std::nullopt, stop, std::nullopt)));
                }
            }
            return true;
        }
        return false;
    };

    if (!handle_colon_prefix()) {
        if (lex.peek_token().kind != Token::Kind::RBracket)
            result.push_back(parse_dim_spec(lex));
    }

    while (lex.peek_token().kind == Token::Kind::Comma) {
        lex.next(); // consume ','
        if (!handle_colon_prefix())
            result.push_back(parse_dim_spec(lex));
    }
    return result;
}

struct DictStep   { std::string key; };
struct VecStep    { std::size_t index; };
struct TensorStep { std::vector<TensorIndex> indices; };
using Step = std::variant<DictStep, VecStep, TensorStep>;
// Parse all [...] steps in slice_info.
std::vector<Step> parse_steps(const std::string& slice_info) {
    std::vector<Step> steps;
    Lexer lex{slice_info};

    while (true) {
        auto tok = lex.next();
        if (tok.kind == Token::Kind::End) break;
        if (tok.kind != Token::Kind::LBracket)
            throw std::runtime_error("gv_slice: expected '[', got: " + tok.text);

        auto inner = lex.peek_token();

        if (inner.kind == Token::Kind::String) {
            // dict key
            lex.next();
            steps.push_back(DictStep{inner.text});
        } else if (inner.kind == Token::Kind::Integer) {
            // might be a plain vector index OR the start of a tensor slice like [0,:,:]
            // We must look ahead: if a comma or colon follows, it's a tensor index.
            // Save position, consume the integer, then peek.
            auto saved = lex.pos;
            lex.next(); // consume integer
            auto nxt2 = lex.peek_token();
            lex.pos = saved; // rewind

            if (nxt2.kind == Token::Kind::Comma || nxt2.kind == Token::Kind::Colon) {
                // tensor multi-index starting with an integer
                steps.push_back(TensorStep{parse_tensor_indices(lex)});
            } else {
                // plain vector/tuple index
                lex.next();
                steps.push_back(VecStep{static_cast<std::size_t>(std::stoul(inner.text))});
            }
        } else {
            // starts with ':' or '...' → must be tensor indices
            steps.push_back(TensorStep{parse_tensor_indices(lex)});
        }

        // consume closing ']'
        auto close = lex.next();
        if (close.kind != Token::Kind::RBracket)
            throw std::runtime_error("gv_slice: expected ']'");
    }
    return steps;
}




}
}