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
import hasty_compute;
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

			Node::TokenReplacer get_cuda_tokreplacer(hasty::dtype dtype) {

				Node::TokenReplacer f32_name_replacer =
					[](int32_t id) 
				{
					switch (id) {
					case DefaultOperatorIDs::POW_ID:
						return "powf";
					case DefaultFunctionIDs::SQRT_ID:
						return "sqrtf";
					case DefaultFunctionIDs::EXP_ID:
						return "expf";
					case DefaultFunctionIDs::LOG_ID:
						return "logf";
					case DefaultFunctionIDs::SIN_ID:
						return "sinf";
					case DefaultFunctionIDs::COS_ID:
						return "cosf";
					case DefaultFunctionIDs::TAN_ID:
						return "tanf";
					case DefaultFunctionIDs::ASIN_ID:
						return "asinf";
					case DefaultFunctionIDs::ACOS_ID:
						return "acosf";
					case DefaultFunctionIDs::ATAN_ID:
						return "atanf";
					case DefaultFunctionIDs::SINH_ID:
						return "sinhf";
					case DefaultFunctionIDs::COSH_ID:
						return "coshf";
					case DefaultFunctionIDs::TANH_ID:
						return "tanhf";
					case DefaultFunctionIDs::ASINH_ID:
						return "asinhf";
					case DefaultFunctionIDs::ACOSH_ID:
						return "acoshf";
					case DefaultFunctionIDs::ATANH_ID:
						return "atanhf";
					default:
						return "";
					}
				};

				Node::TokenReplacer f64_name_replacer =
					[](int32_t id)
				{
					return "";
				};

				if (dtype == hasty::dtype::f32) {
					return f32_name_replacer;
				} else if (dtype == hasty::dtype::f64) {
					return f64_name_replacer;
				}
				throw std::runtime_error("Only non complex floating point types are supported");
			}

			Node::TokenPrinter get_cuda_tokenprinter(hasty::dtype dtype)
			{
				return [](const Node& node) {
					return "";
				};
			}

			/*
			export class CudaWalker : public ExpressionWalker {
			public:

				struct CudaWalkerOptions {
					hasty::dtype dtype;
				};

				std::string walk(const expr::Expression& expr, const CudaWalkerOptions& options) {
					const Node& root = *get_children(expr)[0];
					return walk(root, options);
				}

			private:

				std::string walk(const expr::Node& node, const CudaWalkerOptions& options) {
					switch (node.id()) {
						// Operators
					case DefaultOperatorIDs::NEG_ID:
					{
						return "(-" + walk(*get_children(node)[0], options) + ")";
					}
					case DefaultOperatorIDs::POW_ID:
					{
						if (node.is_zero()) {
							if (options.dtype == hasty::dtype::f32)
								return "(1.0f)";
							else
								return "(1.0)";
						} else if (node.is_unity()) {
							return "(" + walk(*get_children(node)[0], options) + ")";
						} else {
							std::string ret = options.dtype == hasty::dtype::f32 ? "powf(" : "pow(";
							ret += walk(*get_children(node)[0], options) + "," + 
								walk(*get_children(node)[1], options) + ")";
						}
					}
					case DefaultOperatorIDs::MUL_ID:
					{
						const auto& children = get_children(node);
						return "((" + walk(*children[0], options) + ") * (" + walk(*children[1], options) + "))";
					}
					case DefaultOperatorIDs::DIV_ID:
					{
						const auto& children = get_children(node);
						return "((" + walk(*children[0], options) + ") / (" + walk(*children[1], options) + "))";
					}
					case DefaultOperatorIDs::ADD_ID:
					{
						const auto& children = get_children(node);
						return "((" + walk(*children[0], options) + ") + (" + walk(*children[1], options) + "))";
					}
					case DefaultOperatorIDs::SUB_ID:
					{
						const auto& children = get_children(node);
						return "((" + walk(*children[0], options) + ") - (" + walk(*children[1], options) + "))";
					}
					// Unary

					default:
						throw std::runtime_error("Not yet implemented");
					}

				};

			};
			*/

		}
	}

}
