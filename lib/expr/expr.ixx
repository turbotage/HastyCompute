module;

export module expr;

#ifdef STL_AS_MODULES
import std.compat;
#else
import <memory>;
import <stdexcept>;
import <vector>;
import <string>;
import <set>;
import <functional>;
import <deque>;
#endif

import hasty_util;
import token;
import lexer;

namespace hasty {
	namespace expr {

		//export std::unique_ptr<NumberBaseToken> copy_token(const Token& tok);

		export class Expression;

		export class Node {
		public:

			Node(Node&&) = default;

			Node(LexContext& ctext);

			Node(std::vector<std::unique_ptr<Node>>&& childs);

			Node(std::vector<std::unique_ptr<Node>>&& childs, LexContext& ctext);

			Node(std::unique_ptr<NumberBaseToken> base_token, LexContext& ctext);

			virtual std::string str() const = 0;

			virtual std::unique_ptr<Node> copy(LexContext& context) const = 0;

			void fill_variable_list(std::set<std::string>& vars);

			std::unique_ptr<Expression> diff(const std::string& x) const;

			bool child_is_variable(int i) const;

		public:
			LexContext& context;
			std::vector<std::unique_ptr<Node>> children;
			std::unique_ptr<NumberBaseToken> pToken;
		};

		//export std::unique_ptr<Node> node_from_token(const Token& tok, LexContext& context);

		export class TokenNode : public Node {
		public:

			TokenNode(const Token& tok, LexContext& context);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class VariableNode : public Node {
		public:

			VariableNode(const VariableToken& token, LexContext& context);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		private:
			VariableToken m_VarToken;
		};

		export class NegNode : public Node {
		public:

			NegNode(std::unique_ptr<Node> child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class MulNode : public Node {
		public:

			MulNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class DivNode : public Node {
		public:

			DivNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class AddNode : public Node {
		public:

			AddNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class SubNode : public Node {
		public:

			SubNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class PowNode : public Node {
		public:

			PowNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		// UNARY

		export class SgnNode : public Node {
		public:

			SgnNode(std::unique_ptr<Node> child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class AbsNode : public Node {
		public:

			AbsNode(std::unique_ptr<Node> child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class SqrtNode : public Node {
		public:

			SqrtNode(std::unique_ptr<Node> child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class ExpNode : public Node {
		public:

			ExpNode(std::unique_ptr<Node> child)
				: Node(child->context)
			{
				children.emplace_back(std::move(child));
			}

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class LogNode : public Node {
		public:

			LogNode(std::unique_ptr<Node> child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class SinNode : public Node {
		public:

			SinNode(std::unique_ptr<Node> child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class CosNode : public Node {
		public:

			CosNode(std::unique_ptr<Node> child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class TanNode : public Node {
		public:

			TanNode(std::unique_ptr<Node> child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class AsinNode : public Node {
		public:

			AsinNode(std::unique_ptr<Node> child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class AcosNode : public Node {
		public:

			AcosNode(std::unique_ptr<Node> child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class AtanNode : public Node {
		public:

			AtanNode(std::unique_ptr<Node> child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class SinhNode : public Node {
		public:

			SinhNode(std::unique_ptr<Node> child)
				: Node(child->context)
			{
				children.emplace_back(std::move(child));
			}

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class CoshNode : public Node {
		public:

			CoshNode(std::unique_ptr<Node> child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class TanhNode : public Node {
		public:

			TanhNode(std::unique_ptr<Node> child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class AsinhNode : public Node {
		public:

			AsinhNode(std::unique_ptr<Node> child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class AcoshNode : public Node {
		public:

			AcoshNode(std::unique_ptr<Node> child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class AtanhNode : public Node {
		public:

			AtanhNode(std::unique_ptr<Node> child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		};

		export class DerivativeNode : public Node {
		public:

			DerivativeNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		private:

		};

		export class SubsNode : public Node {
		public:

			SubsNode(std::vector<std::unique_ptr<Node>>&& childs);

			std::string str() const override;

			std::unique_ptr<Node> copy(LexContext& context) const override;

		private:

		};

		using ExpressionCreationMap = std::unordered_map<int32_t,
			std::function<void(LexContext&, const Token&, std::vector<std::unique_ptr<Node>>&)>>;

		export class Expression : public Node {
		public:

			Expression(const Expression& other);

			Expression(Expression&&) = default;

			Expression(const std::string& expression, const std::vector<std::string>& variables);

			Expression(const std::string& expression, const LexContext& context);

			Expression(const std::unique_ptr<Node>& root_child, const LexContext& context);

			Expression(const std::unique_ptr<Node>& root_child, const LexContext& context, const std::string& expr);

			Expression(const LexContext& context, const std::deque<std::unique_ptr<Token>>& tokens,
				const ExpressionCreationMap& creation_map);

			std::string str() const override;

			bool is_zero() const;

			std::unique_ptr<Node> copy(LexContext& context) const override;

			static ExpressionCreationMap default_expression_creation_map();

			const LexContext& get_context() const;

			const std::string& get_expression() const;


		private:

			LexContext m_Context;
			std::string m_Expression;
			std::vector<std::string> m_Variables;
		};

		//export Expression expression_creator(const std::string& expression, const LexContext& context);
		//export Expression expression_creator(const std::string& expression, const std::vector<std::string>& variables);

	}
}

