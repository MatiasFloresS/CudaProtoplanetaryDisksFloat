__host__ void UpdateLog (Force *force, PlanetarySystem *sys, float *Dens, float *Energy, int TimeStep, double PhysicalTime,
   int dimfxy);
__host__ Force *AllocateForce (int dimfxy);
__host__ void ComputeForce (Force *fc, float *Dens, double x, double y, double rsmoothing, double mass, int dimfxy);
__host__ double Compute_smoothing (double r);
__host__ void FreeForce (Force *force);
__host__ void Forcescudamalloc (int dimfxy);
