/* Copyright (c) 2008-2017 the MRtrix3 contributors
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * MRtrix is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * For more details, see http://www.mrtrix.org/.
 */


#ifndef __dwi_svr_recon_h__
#define __dwi_svr_recon_h__


#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "types.h"
#include "header.h"
#include "transform.h"
#include "math/SH.h"
#include "dwi/shells.h"
#include "dwi/svr/psf.h"
#include "parallel_for.h"


namespace MR {
  namespace DWI {
    class ReconMatrix;
  }
}


namespace Eigen {
  namespace internal {
    // ReconMatrix inherits its traits from SparseMatrix
    template<>
    struct traits<MR::DWI::ReconMatrix> : public Eigen::internal::traits<Eigen::SparseMatrix<float,Eigen::RowMajor> >
    {};
  }
}


namespace MR
{
  namespace DWI
  {

    class ReconMatrix : public Eigen::EigenBase<ReconMatrix>
    {  MEMALIGN(ReconMatrix);
    public:
      // Required typedefs, constants, and method:
      typedef float Scalar;
      typedef float RealScalar;
      typedef int StorageIndex;
      enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = true
      };

      Eigen::Index rows() const { return nxy*nz*nc; }
      Eigen::Index cols() const { return nxy*nz*nc; }

      template<typename Rhs>
      Eigen::Product<ReconMatrix,Rhs,Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
        return Eigen::Product<ReconMatrix,Rhs,Eigen::AliasFreeProduct>(*this, x.derived());
      }

      typedef Eigen::SparseMatrix<float, Eigen::RowMajor, StorageIndex> SparseMat;
      typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXf;


      // Custom API:
      ReconMatrix(const Header& in, const Eigen::MatrixXf& rigid, const Eigen::MatrixXf& grad, const int lmax, const vector<Eigen::MatrixXf>& rf)
        : lmax (lmax),
          nx (in.size(0)), ny (in.size(1)), nz (in.size(2)), nv (in.size(3)),
          nxy (nx*ny), nc (get_ncoefs(rf)),
          T0 (in),  // Preserve original resolution.
          shellbasis (init_shellbasis(grad, rf)),
          motion (rigid)
      {
        init_Y(grad);
      }


      const RowMatrixXf& getY() const { return Y; }
      const Eigen::MatrixXf& getW() const { return W; }

      void setW (const Eigen::MatrixXf& weights) { W = weights; }

      RowMatrixXf getY0(const Eigen::MatrixXf& grad) const
      {
        DEBUG("initialise Y0");

        vector<size_t> idx = get_shellidx(grad);
        RowMatrixXf Y0 (grad.rows(), nc);

        Eigen::Vector3f vec;
        Eigen::VectorXf delta;

        for (size_t i = 0; i < grad.rows(); i++) {
          vec = {grad(i, 0), grad(i, 1), grad(i, 2)};

          // evaluate basis functions
          Math::SH::delta(delta, vec, lmax);
          Y0.row(i) = shellbasis[idx[i]]*delta;
        }

        return Y0;
      }

      template <typename VectorType1, typename VectorType2>
      void project_x2y(VectorType1& dst, const VectorType2& rhs) const
      {
        DEBUG("Forward projection.");
        size_t nxyz = nxy*nz;
        Eigen::Map<const RowMatrixXf> X (rhs.data(), nxyz, nc);
        Thread::parallel_for<size_t>(0, nv*nz, [&](size_t idx){
          Eigen::VectorXf q = X * Y.row(idx).adjoint();
          Eigen::Ref<Eigen::VectorXf> r = dst.segment(idx*nxy, nxy);
          project_slice_x2y(idx, r, q);
        });
      }

      template <typename VectorType1, typename VectorType2>
      void project_y2x(VectorType1& dst, const VectorType2& rhs) const
      {
        DEBUG("Transpose projection.");
        size_t nxyz = nxy*nz;
        Eigen::Map<RowMatrixXf> X (dst.data(), nxyz, nc);
        Eigen::Map<const Eigen::VectorXf> w (W.data(), W.size());
        RowMatrixXf zero (nxyz, nc); zero.setZero();
        X = Thread::parallel_sum<RowMatrixXf, size_t>(0, nv*nz,
          [&](size_t idx, RowMatrixXf& T){
            Eigen::VectorXf r = Eigen::VectorXf::Zero(nxyz);
            project_slice_y2x(idx, r, rhs.segment(idx*nxy, nxy));
            T.noalias() += w(idx) * r * Y.row(idx);
          }, zero);
      }

      template <typename VectorType1, typename VectorType2>
      void project_x2x(VectorType1& dst, const VectorType2& rhs) const
      {
        DEBUG("Full projection.");
        size_t nxyz = nxy*nz;
        Eigen::Map<const RowMatrixXf> Xi (rhs.data(), nxyz, nc);
        Eigen::Map<RowMatrixXf> Xo (dst.data(), nxyz, nc);
        Eigen::Map<const Eigen::VectorXf> w (W.data(), W.size());
        RowMatrixXf zero (nxyz, nc); zero.setZero();
        Xo = Thread::parallel_sum<RowMatrixXf, size_t>(0, nv*nz,
          [&](size_t idx, RowMatrixXf& T){
            Eigen::VectorXf q = Xi * Y.row(idx).adjoint();
            Eigen::VectorXf r = Eigen::VectorXf::Zero(nxyz);
            project_slice_x2x(idx, r, q);
            T.noalias() += w(idx) * r * Y.row(idx);
        }, zero);
      }


    private:
      const int lmax;
      const size_t nx, ny, nz, nv, nxy, nc;
      const Transform T0;
      const vector<Eigen::MatrixXf> shellbasis;

      RowMatrixXf motion;
      RowMatrixXf Y;
      Eigen::MatrixXf W;


      vector<Eigen::MatrixXf> init_shellbasis(const Eigen::MatrixXf& grad, const vector<Eigen::MatrixXf>& rf) const
      {
        Shells shells (grad.template cast<double>());
        vector<Eigen::MatrixXf> basis;

        for (size_t s = 0; s < shells.count(); s++) {
          Eigen::MatrixXf B;
          if (rf.empty()) {
            B.setIdentity(Math::SH::NforL(lmax), Math::SH::NforL(lmax));
          }
          else {
            B.setZero(nc, Math::SH::NforL(lmax));
            size_t j = 0;
            for (auto& r : rf) {
              for (size_t l = 0; l < r.cols() and 2*l <= lmax; l++)
                for (size_t i = l*(2*l-1); i < (l+1)*(2*l+1); i++, j++)
                  B(j,i) = r(s,l);
            }
          }
          basis.push_back(B);
        }

        return basis;
      }

      void init_Y(const Eigen::MatrixXf& grad)
      {
        DEBUG("initialise Y");
        assert (grad.rows() == nv);     // one gradient per volume

        vector<size_t> idx = get_shellidx(grad);
        Y.resize(nv*nz, nc);

        Eigen::Vector3f vec;
        Eigen::Matrix3f rot;
        rot.setIdentity();
        Eigen::VectorXf delta;

        for (size_t i = 0; i < nv; i++) {
          vec = {grad(i, 0), grad(i, 1), grad(i, 2)};
          if (motion.rows() == nv)
            rot = get_rotation(motion(i,3), motion(i,4), motion(i,5));

          for (size_t j = 0; j < nz; j++) {
            // rotate vector with motion parameters
            if (motion.rows() == nv*nz)
              rot = get_rotation(motion(i*nz+j,3), motion(i*nz+j,4), motion(i*nz+j,5));

            // evaluate basis functions
            Math::SH::delta(delta, rot*vec, lmax);
            Y.row(i*nz+j) = shellbasis[idx[i]]*delta;
          }

        }

      }

      inline Eigen::Matrix3f get_rotation(const float a1, const float a2, const float a3) const
      {
        Eigen::Matrix3f m;
        m = Eigen::AngleAxisf(a1, Eigen::Vector3f::UnitX())
          * Eigen::AngleAxisf(a2, Eigen::Vector3f::UnitY())
          * Eigen::AngleAxisf(a3, Eigen::Vector3f::UnitZ());
        return m;
      }

      inline transform_type get_transform(const Eigen::VectorXf& p) const
      {
        transform_type T;
        T.translation() = Eigen::Vector3d( double(p[0]), double(p[1]), double(p[2]) );
        T.linear() = get_rotation(p[3], p[4], p[5]).template cast<double>();
        return T;
      }

      inline transform_type get_Ts2r(const size_t v, const size_t z) const
      {
        transform_type Ts2r;
        if (motion.rows() == nv) {
          Ts2r = T0.scanner2voxel * get_transform(motion.row(v)) * T0.voxel2scanner;
        } else {
          assert (motion.rows(0) == nv*nz);
          Ts2r = T0.scanner2voxel * get_transform(motion.row(v*nz+z)) * T0.voxel2scanner;
        }
        return Ts2r;
      }

      inline size_t get_idx(const int x, const int y, const int z) const
      {
        return size_t(z*nxy + y*nx + x);
      }

      inline bool inbounds(const int x, const int y, const int z) const
      {
        return (x >= 0) && (x < nx)
            && (y >= 0) && (y < ny)
            && (z >= 0) && (z < nz);
      }

      inline size_t get_ncoefs(const vector<Eigen::MatrixXf>& rf) const
      {
        size_t n = 0;
        if (rf.empty()) {
          n = Math::SH::NforL(lmax);
        } else {
          for (auto& r : rf)
            n += Math::SH::NforL(std::min(2*(int(r.cols())-1), lmax));
        }
        return n;
      }

      vector<size_t> get_shellidx(const Eigen::MatrixXf& grad) const
      {
        Shells shells (grad.template cast<double>());
        vector<size_t> idx (shells.volumecount());

        for (size_t s = 0; s < shells.count(); s++) {
          for (auto v : shells[s].get_volumes())
            idx[v] = s;
        }
        return idx;
      }

      SparseMat get_sliceM(const size_t v, const size_t z) const
      {
        int n = 2;
        SSP<float> ssp (2.0f);

        // reserve memory for elements along each row (outer strides with row-major order).
        SparseMat M (nxy, nxy*nz);
        M.reserve(Eigen::VectorXi::Constant(nxy, (n+1)*8*n*n*n));

        Eigen::Vector3f ps, pr, pg;

        // fill weights
        size_t i = 0;
        float ws, wx, wy, wz, px, py, pz;
        transform_type Ts2r = get_Ts2r(v, z);
        // in-plane
        for (size_t y = 0; y < ny; y++) {
          for (size_t x = 0; x < nx; x++, i++) {

            for (int s = -n; s <= n; s++) {     // ssp neighbourhood
              ws = ssp(s);

              ps = Eigen::Vector3f(x, y, z+s);
              pr = (Ts2r.cast<float>() * ps);
              pg = pr.array().ceil();

              for (int rz = -n; rz < n; rz++) { // local neighbourhood interpolation
                pz = pg[2] + rz;
                if ((pz < 0) || (pz >= nz)) continue;
                wz = bspline<3>(pr[2] - pz);
                for (int ry = -n; ry < n; ry++) {
                  py = pg[1] + ry;
                  if ((py < 0) || (py >= ny)) continue;
                  wy = bspline<3>(pr[1] - py);
                  for (int rx = -n; rx < n; rx++) {
                    px = pg[0] + rx;
                    if ((px < 0) || (px >= nx)) continue;
                    wx = bspline<3>(pr[0] - px);
                    // add to weight matrix.
                    M.coeffRef(i, get_idx(px, py, pz)) += ws * wx * wy * wz;
                  }
                }
              }
            }

          }
        }

        return M;
      }

      Eigen::VectorXf get_sliceY(const size_t v, const size_t z) const
      {
        return Y.row(v*nz+z);
      }

      inline size_t get_grad_idx(const size_t idx) const { return idx / nxy; }
      inline size_t get_grad_idx(const size_t v, const size_t z) const { return v*nz + z; }

      template <typename VectorType1, typename VectorType2>
      void project_slice_x2y(const size_t idx, VectorType1& dst, const VectorType2& rhs) const
      {
        int n = 2;
        SSP<float> ssp (2.0f);
        Eigen::SparseVector<float> m (nxy*nz);
        m.reserve(8*n*n*n);

        Eigen::Vector3f pr;
        size_t v = idx/nz, z = idx%nz;
        float ws;
        transform_type Ts2r = get_Ts2r(v, z);
        for (int s = -n; s <= n; s++) {       // ssp neighbourhood
          ws = ssp(s);
          size_t i = 0;
          for (size_t y = 0; y < ny; y++) {   // in-plane
            for (size_t x = 0; x < nx; x++, i++) {
              pr = Ts2r.cast<float>() * Eigen::Vector3f(x, y, z+s);
              load_sparse_coefs(m, pr);
              dst[i] += ws * m.dot(rhs);
            }
          }
        }
      }

      template <typename VectorType1, typename VectorType2>
      void project_slice_y2x(const size_t idx, VectorType1& dst, const VectorType2& rhs) const
      {
        int n = 2;
        SSP<float> ssp (2.0f);
        Eigen::SparseVector<float> m (nxy*nz);
        m.reserve(8*n*n*n);

        Eigen::Vector3f pr;
        size_t v = idx/nz, z = idx%nz;
        float ws;
        transform_type Ts2r = get_Ts2r(v, z);
        for (int s = -n; s <= n; s++) {       // ssp neighbourhood
          ws = ssp(s);
          size_t i = 0;
          for (size_t y = 0; y < ny; y++) {   // in-plane
            for (size_t x = 0; x < nx; x++, i++) {
              pr = Ts2r.cast<float>() * Eigen::Vector3f(x, y, z+s);
              load_sparse_coefs(m, pr);
              dst += (ws * rhs[i]) * m;
            }
          }
        }
      }

      template <typename VectorType1, typename VectorType2>
      void project_slice_x2x(const size_t idx, VectorType1& dst, const VectorType2& rhs) const
      {
        Eigen::VectorXf tmp = Eigen::VectorXf::Zero(nxy);
        project_slice_x2y(idx, tmp, rhs);
        project_slice_y2x(idx, dst, tmp);
        return;                         // Conservative implementation until FIXME below is sorted.

        // ignored...
        int n = 2;
        SSP<float> ssp (2.0f);
        Eigen::SparseVector<float> m (nxy*nz);
        m.reserve(8*n*n*n);

        Eigen::Vector3f pr;
        size_t v = idx/nz, z = idx%nz;
        float ws, t;
        transform_type Ts2r = get_Ts2r(v, z);
        for (int s = -n; s <= n; s++) {       // ssp neighbourhood
          ws = ssp(s);
          for (size_t y = 0; y < ny; y++) {   // in-plane
            for (size_t x = 0; x < nx; x++) {
              pr = Ts2r.cast<float>() * Eigen::Vector3f(x, y, z+s);
              load_sparse_coefs(m, pr);
              t = ws * m.dot(rhs);      // FIXME: not correct for slice profile...
              dst += (ws * t) * m;
            }
          }
        }
      }

      inline void load_sparse_coefs(Eigen::SparseVector<float>& dst, const Eigen::Vector3f& pr) const
      {
        dst.setZero();
        int n = 2;
        Eigen::Vector3f pg = pr.array().ceil();
        std::array<float,4> wx = bsplineweights<3>(1.0f - (pg[0] - pr[0]));
        std::array<float,4> wy = bsplineweights<3>(1.0f - (pg[1] - pr[1]));
        std::array<float,4> wz = bsplineweights<3>(1.0f - (pg[2] - pr[2]));
        int px, py, pz;
        for (int rz = -n; rz < n; rz++) { // local neighbourhood interpolation
          pz = pg[2] + rz;
          if ((pz < 0) || (pz >= nz)) continue;
          for (int ry = -n; ry < n; ry++) {
            py = pg[1] + ry;
            if ((py < 0) || (py >= ny)) continue;
            for (int rx = -n; rx < n; rx++) {
              px = pg[0] + rx;
              if ((px < 0) || (px >= nx)) continue;
              // insert in weight vector.
              dst.insert(pz*nxy + py*nx + px) += wx[n+rx] * wy[n+ry] * wz[n+rz];
            }
          }
        }
      }


    };


  }
}


// Implementation of ReconMatrix * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen {
  namespace internal {

    template<typename Rhs>
    struct generic_product_impl<MR::DWI::ReconMatrix, Rhs, SparseShape, DenseShape, GemvProduct>
      : generic_product_impl_base<MR::DWI::ReconMatrix,Rhs,generic_product_impl<MR::DWI::ReconMatrix,Rhs> >
    {
      typedef typename Product<MR::DWI::ReconMatrix,Rhs>::Scalar Scalar;

      template<typename Dest>
      static void scaleAndAddTo(Dest& dst, const MR::DWI::ReconMatrix& lhs, const Rhs& rhs, const Scalar& alpha)
      {
        // This method should implement "dst += alpha * lhs * rhs" inplace
        assert(alpha==Scalar(1) && "scaling is not implemented");

        lhs.project_x2x<Dest, Rhs>(dst, rhs);

      }

    };
  }
}


#endif

