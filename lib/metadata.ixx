module;

export module metadata;

import <memory>;
import <stdexcept>;
import <vector>;
import <string>;
import <unordered_map>;
import <complex>;

import hasty_util;

namespace hasty {

	export class Metadata {
	public:

		enum class Type {
			eNumber,
			eString
		};

		Metadata::Type get_type() { return _type; }

		Metadata& insert(const std::string& name, const sptr<Metadata>& mdata)
		{
			_contents.insert_or_assign(name, mdata); 
			return *this;
		}

		Metadata& operator[](const std::string& in) { return *_contents[in]; }

	protected:

		Metadata(Metadata::Type type) : _type(type) {}

		Metadata::Type _type;

	private:

		std::unordered_map<std::string, sptr<Metadata>> _contents;

	};

	export class MetadataNumber : public Metadata {
	public:

		MetadataNumber(std::complex<float> in) 
			: Metadata(Metadata::Type::eNumber)
		{
			cf64 = in;	
			_numericType = MetadataNumber::Type::cf64;
		}
		MetadataNumber(std::complex<double> in)
			: Metadata(Metadata::Type::eNumber)
		{
			cf128 = in;	
			_numericType = MetadataNumber::Type::cf128;
		}
		MetadataNumber(float in)
			: Metadata(Metadata::Type::eNumber)
		{ 
			f32 = in;		
			_numericType = MetadataNumber::Type::f32;
		}
		MetadataNumber(double in)				
			: Metadata(Metadata::Type::eNumber)
		{ 
			f64 = in;		
			_numericType = MetadataNumber::Type::f64;
		}
		MetadataNumber(i32 in)
			: Metadata(Metadata::Type::eNumber)
		{
			i32 = in;
			_numericType = MetadataNumber::Type::i32;
		}
		MetadataNumber(i64 in)
			: Metadata(Metadata::Type::eNumber)
		{
			i64 = in;
			_numericType = MetadataNumber::Type::i64;
		}

		enum class Type {
			cf64,
			cf128,
			f32,
			f64,
			i32,
			i64
		};

		MetadataNumber::Type get_numeric_type() { return _numericType; }

		std::complex<float>		get_cf64()	{ return cf64;	}
		std::complex<double>	get_cf128() { return cf128; }
		float					get_f32()	{ return f32;	}
		double					get_f64()	{ return f64;	}
		i32						get_i32()	{ return i32;	}
		i64						get_i64()	{ return i64;	}

		MetadataNumber& set_numeric(std::complex<float> in)
		{ 
			cf64 = in;
			_numericType = MetadataNumber::Type::cf64;
			return *this;
		}
		MetadataNumber& set_numeric(std::complex<double> in)
		{
			cf128 = in;
			_numericType = MetadataNumber::Type::cf128;
			return *this;
		}
		MetadataNumber& set_numeric(float in)
		{
			f32 = in;
			_numericType = MetadataNumber::Type::f32;
			return *this;
		}
		MetadataNumber& set_numeric(double in)
		{
			f64 = in;
			_numericType = MetadataNumber::Type::f64;
			return *this;
		}
		MetadataNumber& set_numeric(i32 in)
		{
			i32 = in;
			_numericType = MetadataNumber::Type::i32;
			return *this;
		}
		MetadataNumber& set_numeric(i64 in)
		{
			i64 = in;
			_numericType = MetadataNumber::Type::i64;
			return *this;
		}


	private:

		MetadataNumber::Type _numericType;

		union {
			std::complex<float>		cf64;
			std::complex<double>	cf128;
			float					f32;
			double					f64;
			i32						i32;
			i64						i64;
		};

	};

	export class MetadataString : public Metadata {
	public:

		MetadataString(const std::string& in)
			: Metadata(Metadata::Type::eString), _str(in) {}
		
		std::string get_str() { return _str; }
		MetadataString& set_str(const std::string& str) 
		{ 
			_str = str;
			return *this;
		}

	private:

		std::string _str;

	};

}