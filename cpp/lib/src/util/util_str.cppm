module;

export module util_mod:str;

import std;

namespace hasty {

struct TabbedWriter {
    std::string m_buffer;
    std::int32_t m_tab_level;

    TabbedWriter() : m_tab_level(0) {}

    void add_reserved(std::int32_t n_chars) {
        m_buffer.reserve(m_buffer.size() + n_chars);
    }

    void tab_in() {
        m_tab_level++;
    }

    void tab_out() {
        if (m_tab_level > 0) {
            m_tab_level--;
        }
    }

    bool on_newline() {
        if (m_buffer.back() == '\n') {
            return true;
        }
        return false;
    }

    void write_line(const std::string& line) {
        m_buffer.append(std::string(m_tab_level * 4, '\t'));
        m_buffer.append(line);
        m_buffer.append("\n");
    }

    void append_string_tabbed(const std::string& other) {
        std::string line;
        line.reserve(other.size());
        for (const auto& c : other) {
            if (c == '\n') {
                write_line(line);
                line.clear();
            } else {
                line.push_back(c);
            }
        }
        if (!line.empty()) {
            write_line(line);
        }
    }

    std::string str() const {
        return m_buffer;
    }

};

}