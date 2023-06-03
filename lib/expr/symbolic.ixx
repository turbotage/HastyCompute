module;

#include <symengine/expression.h>
#include <symengine/simplify.h>
#include <symengine/parser.h>

export module symbolic;

#ifdef STL_AS_MODULES
import std.compat;
#else
import <memory>;
import <stdexcept>;
import <vector>;
import <string>;
#endif

import expr;
import defaultexp;
import hasty_util;
import metadata;

namespace hasty {

	namespace expr {

		export class SymbolicVariable : public MetadataString {
		public:
			 
			SymbolicVariable(const std::string& name)
				: MetadataString(name) {}
			
			enum class Type {
				eStatic,
				eVarying
			};

			SymbolicVariable(const std::string& name, SymbolicVariable::Type symtype)
				: MetadataString(name), _symType(symtype) {}

			SymbolicVariable::Type sym_type() const { return _symType; }

			SymbolicVariable::Type& sym_type() { return _symType; }

		private:

			SymbolicVariable::Type _symType;

		};

		export class SymbolicContext {
		public:

			void insert_variable(const SymbolicVariable& var)
			{
				_vars.push_back(var);
			}

			void insert_variable(const std::string& name)
			{
				_vars.emplace_back(name);
			}

			void insert_variable(const std::string& name, SymbolicVariable::Type symtype)
			{
				_vars.emplace_back(name, symtype);
			}

			SymbolicVariable& operator[](const std::string& name)
			{
				for (auto& var : _vars) {
					if (var.get_str() == name) {
						return var;
					}
				}
				throw std::runtime_error("SymbolicContext did not contain variable with name" + name);
			}

			vec<SymbolicVariable>::const_iterator begin() const
			{
				return _vars.begin();
			}

			vec<SymbolicVariable>::const_iterator end() const
			{
				return _vars.end();
			}

		private:

			std::vector<SymbolicVariable> _vars;

		};

		namespace cuda {

			export class CudaWalker : public ExpressionWalker {
			public:

				std::string walk(const expr::Expression& expr) {
					const Node& root = *get_children(expr)[0];
					return walk(root);
				}

			private:

				std::string walk(const expr::Node& node) {
					switch (node.id()) {
					// Operators
					case DefaultOperatorIDs::NEG_ID:
					{
						return "(-" + walk(*get_children(node)[0]) + ")";
					}
					case DefaultOperatorIDs::POW_ID:
					{
						throw std::runtime_error("POW not yet implemented");
					}
					case DefaultOperatorIDs::MUL_ID:
					{
						const auto& children = get_children(node);
						return "((" + walk(*children[0]) + ") * (" + walk(*children[1]) + "))";
					}
					case DefaultOperatorIDs::DIV_ID:
					{
						const auto& children = get_children(node);
						return "((" + walk(*children[0]) + ") / (" + walk(*children[1]) + "))";
					}
					case DefaultOperatorIDs::ADD_ID:
					{
						const auto& children = get_children(node);
						return "((" + walk(*children[0]) + ") + (" + walk(*children[1]) + "))";
					}
					case DefaultOperatorIDs::SUB_ID:
					{
						const auto& children = get_children(node);
						return "((" + walk(*children[0]) + ") - (" + walk(*children[1]) + "))";
					}
					// Unary
					default:
						throw std::runtime_error("Not yet implemented");
				}

			};

		}

	}

}
