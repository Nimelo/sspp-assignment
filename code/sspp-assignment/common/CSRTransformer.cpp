#include "CSRTransformer.h"
#include "Definitions.h"

representations::csr::CSR tools::transformers::csr::CSRTransformer::transform(representations::intermediary::IntermediarySparseMatrix ism)
{
	FLOATING_TYPE *AS = new FLOATING_TYPE[ism.getNZ()];
	int index = 0, *IRP = new int[ism.getM() + 1], *JA = new int[ism.getNZ()];
	
	AS[0] = ism.getValues()[0];
	JA[0] = ism.getJIndexes()[0];
	IRP[0] = 0;

	for (int i = 1; i < ism.getNZ(); i++)
	{
		AS[i] = ism.getValues()[i];
		JA[i] = ism.getJIndexes()[i];
		if (ism.getIIndexes()[i - 1] != ism.getIIndexes()[i])
			IRP[++index] = i;
	}

	IRP[++index] = ism.getNZ();

	return representations::csr::CSR(ism.getNZ(), ism.getM(), ism.getN(), IRP, JA, AS);
}
