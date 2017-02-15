#ifndef __H_FLOPS_CALCULATOR
#define __H_FLOPS_CALCULATOR

namespace tools
{
	namespace measurements
	{
		namespace calculators
		{
			class FLOPSCalculator
			{
				public:
					static double calculate(int nz, double miliseconds);
					static double calculate(int nz, long miliseconds);
			};
		}
	}
}

#endif