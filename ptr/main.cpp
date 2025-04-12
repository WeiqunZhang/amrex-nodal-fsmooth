#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_ParmParse.H>

#define NTHREADS 128

using namespace amrex;

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int mlndlap_color (int i, int j, int k)
{
    return (i%2) + (j%2)*2 + (k%2)*4;
}

#define IDX(i, j, k) (i) + (j)*nx + (k)*nx*ny

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlndlap_gscolor_c (int i, int j, int k, int nx, int ny, int nz,
                        Real * AMREX_RESTRICT sol,
                        Real const* AMREX_RESTRICT rhs, Real sig,
                        int const* AMREX_RESTRICT msk,
                        GpuArray<Real,AMREX_SPACEDIM> const& dxinv, int color) noexcept
{
  if ( i>0 && i <(nx-1) && j>0 && j<(ny-1) && k>0 && k < (nz-1) ) {
    if (mlndlap_color(i,j,k) == color) {
        if (msk[IDX(i,j,k)]) {
            sol[IDX(i,j,k)] = Real(0.0);
        } else {
            Real facx = Real(1.0/36.0)*dxinv[0]*dxinv[0];
            Real facy = Real(1.0/36.0)*dxinv[1]*dxinv[1];
            Real facz = Real(1.0/36.0)*dxinv[2]*dxinv[2];
            Real fxyz = facx + facy + facz;
            Real fmx2y2z = -facx + Real(2.0)*facy + Real(2.0)*facz;
            Real f2xmy2z = Real(2.0)*facx - facy + Real(2.0)*facz;
            Real f2x2ymz = Real(2.0)*facx + Real(2.0)*facy - facz;
            Real f4xm2ym2z = Real(4.0)*facx - Real(2.0)*facy - Real(2.0)*facz;
            Real fm2x4ym2z = -Real(2.0)*facx + Real(4.0)*facy - Real(2.0)*facz;
            Real fm2xm2y4z = -Real(2.0)*facx - Real(2.0)*facy + Real(4.0)*facz;

            Real s0 = Real(-4.0)*fxyz*Real(8.);
            Real Ax = sol[IDX(i,j,k)]*s0
                + fxyz*(sol[IDX(i-1,j-1,k-1)]
                      + sol[IDX(i+1,j-1,k-1)]
                      + sol[IDX(i-1,j+1,k-1)]
                      + sol[IDX(i+1,j+1,k-1)]
                      + sol[IDX(i-1,j-1,k+1)]
                      + sol[IDX(i+1,j-1,k+1)]
                      + sol[IDX(i-1,j+1,k+1)]
                      + sol[IDX(i+1,j+1,k+1)])
                + fmx2y2z*(sol[IDX(i  ,j-1,k-1)]*Real(2.)
                         + sol[IDX(i  ,j+1,k-1)]*Real(2.)
                         + sol[IDX(i  ,j-1,k+1)]*Real(2.)
                         + sol[IDX(i  ,j+1,k+1)]*Real(2.))
                + f2xmy2z*(sol[IDX(i-1,j  ,k-1)]*Real(2.)
                         + sol[IDX(i+1,j  ,k-1)]*Real(2.)
                         + sol[IDX(i-1,j  ,k+1)]*Real(2.)
                         + sol[IDX(i+1,j  ,k+1)]*Real(2.))
                + f2x2ymz*(sol[IDX(i-1,j-1,k  )]*Real(2.)
                         + sol[IDX(i+1,j-1,k  )]*Real(2.)
                         + sol[IDX(i-1,j+1,k  )]*Real(2.)
                         + sol[IDX(i+1,j+1,k  )]*Real(2.))
                + f4xm2ym2z*(sol[IDX(i-1,j,k)]*Real(4.)
                           + sol[IDX(i+1,j,k)]*Real(4.))
                + fm2x4ym2z*(sol[IDX(i,j-1,k)]*Real(4.)
                           + sol[IDX(i,j+1,k)]*Real(4.))
                + fm2xm2y4z*(sol[IDX(i,j,k-1)]*Real(4.)
                           + sol[IDX(i,j,k+1)]*Real(4.));

            sol[IDX(i,j,k)] += (rhs[IDX(i,j,k)] - Ax*sig) / (s0*sig);
        }
    }
  }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    amrex::SetVerbose(0);
    {
        int n_cell = 128;
        ParmParse pp;
        pp.query("n_cell", n_cell);

        int nx = n_cell;
        int ny = n_cell;
        int nz = n_cell;

	Box box(IntVect(0), IntVect(n_cell-1));

	FArrayBox solfab(box, 1);
	FArrayBox rhsfab(box, 1);
	IArrayBox mskfab(box, 1);
	solfab.template setVal<RunOn::Device>(1.0);
	rhsfab.template setVal<RunOn::Device>(2.0);
	mskfab.template setVal<RunOn::Device>(0);

	Real sig = 1.0;
	GpuArray<Real,3> dxinv{1.0,1.0,1.0};

	Real* sol = solfab.dataPtr();
	Real const* rhs = rhsfab.dataPtr();
	int const* msk = mskfab.dataPtr();

	for (int color = 0; color < 8; ++color) {
	    ParallelFor<NTHREADS>(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
		mlndlap_gscolor_c(i,j,k,nx,ny,nz,sol,rhs,sig,msk,dxinv,color);
	    });
	}
	Gpu::streamSynchronize();

	double t0 = second();

	int iterations = 10;
	for (int it = 0; it < iterations; ++it) {
	    for (int color = 0; color < 8; ++color) {
		ParallelFor<NTHREADS>(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
		    mlndlap_gscolor_c(i,j,k,nx,ny,nz,sol,rhs,sig,msk,dxinv,color);
		});
	    }
	    Gpu::streamSynchronize();
	}
	
	double t1 = second();
	std::cout << "Run time: " << t1-t0 << "\n";
    }
    amrex::Finalize();
}
