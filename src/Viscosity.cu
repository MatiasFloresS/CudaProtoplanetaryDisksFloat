#include "Main.cuh"

extern int NSEC, size_grid, NRAD;

extern float TRANSITIONWIDTH, TRANSITIONRADIUS, TRANSITIONRATIO, ASPECTRATIO, LAMBDADOUBLING;
extern float VISCOSITY, CAVITYRATIO, CAVITYRADIUS, CAVITYWIDTH, ALPHAVISCOSITY;
extern float ViscosityAlpha, onethird, invdphi;

extern float *SoundSpeed, *GLOBAL_bufarray, *viscosity_array, *invRmed, *Rmed, *Rsup, *VradInt, *VthetaInt;
extern float *SoundSpeed_d, *viscosity_array_d, *invdiffRmed_d, *Rinf_d, *invRinf_d, *invRmed_d, *Rmed_d;
extern float *invdiffRsup_d, *Rsup_d, *Dens_d, *Vrad_d, *Vtheta_d, *Vradial_d, *Vazimutal_d, *VthetaInt_d, *VradInt_d;

float *DivergenceVelocity, *DRP, *DRR, *DPP, *TAURR, *TAURP, *TAUPP;
float *DivergenceVelocity_d, *DRP_d, *DRR_d, *DPP_d, *TAURR_d, *TAURP_d, *TAUPP_d;
float PhysicalTime =0.0, PhysicalTimeInitial= 0.0;

extern dim3 dimGrid2, dimBlock2;

__host__ float FViscosity(float r)
{
  float viscosity, rmin, rmax, scale;
  int i = 0;
  viscosity = VISCOSITY;
  if (ViscosityAlpha){
    while (Rmed[i] < r) i++;
    viscosity = ALPHAVISCOSITY*GLOBAL_bufarray[i] * GLOBAL_bufarray[i] * pow(r, 1.5);
  }
  rmin = CAVITYRADIUS-CAVITYWIDTH*ASPECTRATIO;
  rmax = CAVITYRADIUS+CAVITYWIDTH*ASPECTRATIO;
  scale = 1.0+(PhysicalTime-PhysicalTimeInitial)*LAMBDADOUBLING;
  rmin *= scale;
  rmax *= scale;
  if (r < rmin) viscosity *= CAVITYRATIO;
  if ((r >= rmin) && (r <= rmax))
    viscosity *= exp((rmax-r)/(rmax-rmin)*log(CAVITYRATIO));
  return viscosity;
}

__host__ float AspectRatioHost(float r)
{
  float aspectratio, rmin, rmax, scale;
  aspectratio = ASPECTRATIO;
  rmin = TRANSITIONRADIUS-TRANSITIONWIDTH*ASPECTRATIO;
  rmax = TRANSITIONRADIUS+TRANSITIONWIDTH*ASPECTRATIO;
  scale = 1.0+(PhysicalTime-PhysicalTimeInitial)*LAMBDADOUBLING;
  rmin *= scale;
  rmax *= scale;
  if (r < rmin) aspectratio *= TRANSITIONRATIO;
  if ((r >= rmin) && (r <= rmax))
    aspectratio *= exp((rmax-r)/(rmax-rmin)*log(TRANSITIONRATIO));
  return aspectratio;
}

__host__ void InitViscosity ()
{
  DivergenceVelocity  = (float *)malloc(size_grid*sizeof(float));
  DRR                 = (float *)malloc(size_grid*sizeof(float));
  DRP                 = (float *)malloc(size_grid*sizeof(float));
  DPP                 = (float *)malloc(size_grid*sizeof(float));
  TAURR               = (float *)malloc(size_grid*sizeof(float));
  TAURP               = (float *)malloc(size_grid*sizeof(float));
  TAUPP               = (float *)malloc(size_grid*sizeof(float));
  InitViscosityDevice ();
}

__host__ void InitViscosityDevice ()
{
  gpuErrchk(cudaMalloc((void**)&DivergenceVelocity_d, size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&DRR_d,                size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&DRP_d,                size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&DPP_d,                size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&TAURR_d,              size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&TAURP_d,              size_grid*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&TAUPP_d,              size_grid*sizeof(float)));
}

__host__ void ComputeViscousTerms (float *Vradial_d, float *Vazimutal_d, float *Dens)
{

  if (ViscosityAlpha){
    //gpuErrchk(cudaMemcpy(SoundSpeed, SoundSpeed_d, size_grid*sizeof(float), cudaMemcpyDeviceToHost));
    Make1Dprofile (1);
  }

  for (int i = 0; i < NRAD; i++)
    viscosity_array[i] = FViscosity(Rmed[i]);
  gpuErrchk(cudaMemcpy(viscosity_array_d, viscosity_array, (NRAD+1)*sizeof(float), cudaMemcpyHostToDevice));

  ViscousTermsKernelDRP<<<dimGrid2, dimBlock2>>>(Vradial_d, Vazimutal_d , DRR_d, DPP_d, DivergenceVelocity_d,
    DRP_d, invdiffRsup_d, invRmed_d, Rsup_d, Rinf_d, invdiffRmed_d, NRAD, NSEC, invRinf_d, invdphi);
  gpuErrchk(cudaDeviceSynchronize());

  ViscousTermsKernelTAURP<<<dimGrid2, dimBlock2>>>(Dens_d, viscosity_array_d, DRR_d, DPP_d, onethird, DivergenceVelocity_d,
    TAURR_d, TAUPP_d, TAURP_d, DRP_d, NRAD, NSEC);
  gpuErrchk(cudaDeviceSynchronize());

}

__host__ void UpdateVelocitiesWithViscosity(float *VradInt, float *VthetaInt, float *Dens, float DeltaT)
{
  UpdateVelocitiesKernel<<<dimGrid2, dimBlock2>>>(VthetaInt_d, VradInt_d, invRmed_d, Rmed_d, Rsup_d, Rinf_d,
    invdiffRmed_d, invdiffRsup_d,  Dens_d, invRinf_d, TAURR_d, TAURP_d, TAUPP_d, DeltaT, NRAD, NSEC, invdphi);
    gpuErrchk(cudaDeviceSynchronize());
}
