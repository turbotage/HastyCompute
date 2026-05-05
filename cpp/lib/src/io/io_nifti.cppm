module;

#include <nifti1_io.h>
#include <cstring>
#include <cstdlib>

export module hasty_io_mod:nifti;

import std;
import hasty_util_mod;
import hasty_tensor_mod;

namespace hasty {
namespace io {
namespace nifti {

export struct NiftiHeader {
    std::array<f32, 8>                pixdim;
    std::array<std::array<f32, 4>, 4> qform;
    std::array<std::array<f32, 4>, 4> sform;
    i32         qform_code;
    i32         sform_code;
    f32         scl_slope;
    f32         scl_inter;
    std::string description;
};

export struct NiftiImage {
    Tensor      data;
    NiftiHeader header;
};

namespace {

eScalarType nifti_type_to_scalar(int dt)
{
    switch (dt) {
        case DT_UINT8:       return scalar_alias::u8;
        case DT_INT8:        return scalar_alias::i8;
        case DT_INT16:       return scalar_alias::i16;
        case DT_INT32:       return scalar_alias::i32;
        case DT_INT64:       return scalar_alias::i64;
        case DT_FLOAT32:     return scalar_alias::f32;
        case DT_FLOAT64:     return scalar_alias::f64;
        case DT_COMPLEX64:   return scalar_alias::c32;
        case DT_COMPLEX128:  return scalar_alias::c64;
        default:
            throw std::runtime_error("[nifti] Unsupported datatype: " + std::to_string(dt));
    }
}

int scalar_to_nifti_type(eScalarType st)
{
    switch (st) {
        case eScalarType::Byte:          return DT_UINT8;
        case eScalarType::Char:          return DT_INT8;
        case eScalarType::Short:         return DT_INT16;
        case eScalarType::Int:           return DT_INT32;
        case eScalarType::Long:          return DT_INT64;
        case eScalarType::Float:         return DT_FLOAT32;
        case eScalarType::Double:        return DT_FLOAT64;
        case eScalarType::ComplexFloat:  return DT_COMPLEX64;
        case eScalarType::ComplexDouble: return DT_COMPLEX128;
        default:
            throw std::runtime_error("[nifti] No NIfTI mapping for scalar type: " +
                                     scalar_type_to_string(st));
    }
}

} // namespace

export NiftiImage read_nifti(const std::string& filename)
{
    nifti_image* nim = ::nifti_image_read(filename.c_str(), 1);
    if (!nim)
        throw std::runtime_error("[nifti read] Failed to read: " + filename);

    eScalarType dtype = nifti_type_to_scalar(nim->datatype);

    std::vector<i64> shape;
    int raw_dims[7] = { nim->nx, nim->ny, nim->nz, nim->nt, nim->nu, nim->nv, nim->nw };
    for (int d = 0; d < nim->ndim; ++d)
        shape.push_back(static_cast<i64>(raw_dims[d]));

    size_t nbytes = static_cast<size_t>(nim->nvox) * static_cast<size_t>(nim->nbyper);
    std::vector<u8> bytes(nbytes);
    std::memcpy(bytes.data(), nim->data, nbytes);

    NiftiHeader hdr;
    for (int i = 0; i < 8; ++i)
        hdr.pixdim[i] = nim->pixdim[i];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            hdr.qform[i][j] = nim->qto_xyz.m[i][j];
            hdr.sform[i][j] = nim->sto_xyz.m[i][j];
        }
    hdr.qform_code  = nim->qform_code;
    hdr.sform_code  = nim->sform_code;
    hdr.scl_slope   = nim->scl_slope;
    hdr.scl_inter   = nim->scl_inter;
    hdr.description = nim->descrip;

    ::nifti_image_free(nim);

    return NiftiImage{
        Tensor::from_vector(std::move(bytes), shape, dtype, Device()),
        std::move(hdr)
    };
}

export void write_nifti(const NiftiImage& img, const std::string& filename)
{
    if (img.data.device().type != eDeviceType::CPU)
        throw std::runtime_error("[nifti write] Tensor must be on CPU");
    if (!img.data.is_contiguous())
        throw std::runtime_error("[nifti write] Tensor must be contiguous");

    eScalarType dtype    = img.data.scalar_type();
    int         nifti_dt = scalar_to_nifti_type(dtype);
    int         nbyper, swapsize;
    ::nifti_datatype_sizes(nifti_dt, &nbyper, &swapsize);

    ArrayRef<i64> sizes = img.data.sizes();
    int ndim = static_cast<int>(sizes.size());

    nifti_image* nim = static_cast<nifti_image*>(std::calloc(1, sizeof(nifti_image)));
    if (!nim)
        throw std::runtime_error("[nifti write] Allocation failed");

    nim->fname      = ::nifti_strdup(filename.c_str());
    nim->iname      = ::nifti_strdup(filename.c_str());
    nim->nifti_type = NIFTI_FTYPE_NIFTI1_1;
    nim->ndim       = ndim;
    nim->datatype   = nifti_dt;
    nim->nbyper     = nbyper;
    nim->swapsize   = swapsize;

    // native byte order
    union { unsigned char b[2]; short s; } bcheck;
    bcheck.s = 1;
    nim->byteorder  = (bcheck.b[0] == 1) ? 1 : 2; // 1=LSB_FIRST, 2=MSB_FIRST

    int* dim_ptrs[7] = { &nim->nx, &nim->ny, &nim->nz,
                         &nim->nt, &nim->nu, &nim->nv, &nim->nw };
    nim->dim[0] = ndim;
    nim->nvox   = 1;
    for (int d = 0; d < ndim; ++d) {
        *dim_ptrs[d]  = static_cast<int>(sizes[d]);
        nim->dim[d+1] = static_cast<int>(sizes[d]);
        nim->nvox    *= static_cast<size_t>(sizes[d]);
    }

    for (int i = 0; i < 8; ++i)
        nim->pixdim[i] = img.header.pixdim[i];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            nim->qto_xyz.m[i][j] = img.header.qform[i][j];
            nim->sto_xyz.m[i][j] = img.header.sform[i][j];
        }
    nim->qform_code = img.header.qform_code;
    nim->sform_code = img.header.sform_code;
    nim->scl_slope  = img.header.scl_slope;
    nim->scl_inter  = img.header.scl_inter;

    std::string desc = img.header.description.substr(0, 79);
    std::memcpy(nim->descrip, desc.c_str(), desc.size());

    size_t nbytes = nim->nvox * static_cast<size_t>(nbyper);
    nim->data = std::malloc(nbytes);
    if (!nim->data) {
        ::nifti_image_free(nim);
        throw std::runtime_error("[nifti write] Data allocation failed");
    }
    std::memcpy(nim->data, img.data.const_data_ptr(), nbytes);

    ::nifti_image_write(nim);
    ::nifti_image_free(nim);
}

}
}
}
