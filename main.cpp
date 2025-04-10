#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int mlndlap_color (int i, int j, int k)
{
    return (i%2) + (j%2)*2 + (k%2)*4;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlndlap_gscolor_c (int i, int j, int k, Array4<Real> const& sol,
                        Array4<Real const> const& rhs, Real sig,
                        Array4<int const> const& msk,
                        GpuArray<Real,AMREX_SPACEDIM> const& dxinv, int color) noexcept
{
    if (mlndlap_color(i,j,k) == color) {
        if (msk(i,j,k)) {
            sol(i,j,k) = Real(0.0);
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
            Real Ax = sol(i,j,k)*s0
                + fxyz*(sol(i-1,j-1,k-1)
                      + sol(i+1,j-1,k-1)
                      + sol(i-1,j+1,k-1)
                      + sol(i+1,j+1,k-1)
                      + sol(i-1,j-1,k+1)
                      + sol(i+1,j-1,k+1)
                      + sol(i-1,j+1,k+1)
                      + sol(i+1,j+1,k+1))
                + fmx2y2z*(sol(i  ,j-1,k-1)*Real(2.)
                         + sol(i  ,j+1,k-1)*Real(2.)
                         + sol(i  ,j-1,k+1)*Real(2.)
                         + sol(i  ,j+1,k+1)*Real(2.))
                + f2xmy2z*(sol(i-1,j  ,k-1)*Real(2.)
                         + sol(i+1,j  ,k-1)*Real(2.)
                         + sol(i-1,j  ,k+1)*Real(2.)
                         + sol(i+1,j  ,k+1)*Real(2.))
                + f2x2ymz*(sol(i-1,j-1,k  )*Real(2.)
                         + sol(i+1,j-1,k  )*Real(2.)
                         + sol(i-1,j+1,k  )*Real(2.)
                         + sol(i+1,j+1,k  )*Real(2.))
                + f4xm2ym2z*(sol(i-1,j,k)*Real(4.)
                           + sol(i+1,j,k)*Real(4.))
                + fm2x4ym2z*(sol(i,j-1,k)*Real(4.)
                           + sol(i,j+1,k)*Real(4.))
                + fm2xm2y4z*(sol(i,j,k-1)*Real(4.)
                           + sol(i,j,k+1)*Real(4.));

            sol(i,j,k) += (rhs(i,j,k) - Ax*sig) / (s0*sig);
        }
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    amrex::SetVerbose(0);
    {
        int n_cell = 256;
        ParmParse pp;
        pp.query("n_cell", n_cell);

	Box box(IntVect(0), IntVect(n_cell-1));

	FArrayBox solfab(amrex::grow(box,1), 1);
	FArrayBox rhsfab(box, 1);
	IArrayBox mskfab(box, 1);
	solfab.template setVal<RunOn::Device>(1.0);
	rhsfab.template setVal<RunOn::Device>(2.0);
	mskfab.template setVal<RunOn::Device>(0);

	Real sig = 1.0;
	GpuArray<Real,3> dxinv{1.0,1.0,1.0};

	auto const& sol = solfab.array();
	auto const& rhs = rhsfab.const_array();
	auto const& msk = mskfab.const_array();

	for (int color = 0; color < 8; ++color) {
	    ParallelFor<128>(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
		mlndlap_gscolor_c(i,j,k,sol,rhs,sig,msk,dxinv,color);
	    });
	}
	Gpu::streamSynchronize();

	double t0 = second();

	int iterations = 10;
	for (int it = 0; it < iterations; ++it) {
	    for (int color = 0; color < 8; ++color) {
		ParallelFor<128>(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
		    mlndlap_gscolor_c(i,j,k,sol,rhs,sig,msk,dxinv,color);
		});
	    }
	    Gpu::streamSynchronize();
	}
	
	double t1 = second();
	std::cout << "Run time: " << t1-t0 << "\n";
    }
    amrex::Finalize();
}
