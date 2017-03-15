#include "ELLPACKTransformer.h"
#include "InPlaceStableSorter.h"
#include <algorithm>

void sspp::tools::transformers::ELLPACKTransformer::preprocessISM(const representations::IntermediarySparseMatrix & ism)
{
	tools::sorters::InPlaceStableSorter sorter;
	sorter.sort(ism.IIndexes, ism.JIndexes, ism.Values, ism.NZ);
}

int * sspp::tools::transformers::ELLPACKTransformer::findAuxilliaryArray(const representations::IntermediarySparseMatrix & ism)
{
	int * auxArray = new int[ism.M];

	auto tmp = 0;
	auto index = 0;
	for (auto i = 1; i < ism.NZ; i++)
	{
		if (ism.IIndexes[i - 1] == ism.IIndexes[i])
		{
			++tmp;
		}
		else
		{
			auxArray[index++] = ++tmp;
			tmp = 0;
		}
	}

	auxArray[index++] = tmp != 0 ? ++tmp : 1;
	return auxArray;
}

void sspp::tools::transformers::ELLPACKTransformer::allocateArrays(int *** JA, FLOATING_TYPE *** AS, int M, int MAXNZ)
{
	*JA = new int*[M];
	for (auto i = 0; i < M; i++)
		(*JA)[i] = new int[MAXNZ];

	*AS = new FLOATING_TYPE*[M];
	for (auto i = 0; i < M; i++)
		(*AS)[i] = new FLOATING_TYPE[MAXNZ];
}

sspp::representations::ELLPACK sspp::tools::transformers::ELLPACKTransformer::transformImpl(const representations::IntermediarySparseMatrix & ism, int M, int MAXNZ, int ** JA, FLOATING_TYPE ** AS, int * auxArray)
{
	auto nzIndex = 0;
	for (auto row = 0; row < M; row++)
	{
		for (auto column = 0; column < auxArray[row]; column++)
		{
			AS[row][column] = ism.Values[nzIndex];
			JA[row][column] = ism.JIndexes[nzIndex];
			++nzIndex;
		}

		if (auxArray[row] < MAXNZ)
		{
			for (auto i = auxArray[row]; i < MAXNZ; i++)
			{
				AS[row][i] = 0;
				if (auxArray[row] != 0)
					JA[row][i] = JA[row][i - 1];
				else
					JA[row][i] = 0;
			}
		}
	}
	return representations::ELLPACK(M, ism.N, ism.NZ, MAXNZ, JA, AS);
}

sspp::representations::ELLPACK sspp::tools::transformers::ELLPACKTransformer::transform(const representations::IntermediarySparseMatrix & ism)
{
	preprocessISM(ism);
	int *aux = findAuxilliaryArray(ism);
	auto MAXNZ = *std::max_element(aux, aux + ism.M - 1);

	int **JA = NULL;
	double **AS = NULL;
	allocateArrays(&JA, &AS, ism.M, MAXNZ);
	
	auto ellpack = transformImpl(ism, ism.M, MAXNZ, JA, AS, aux);
	delete[] aux;

	return ellpack;
}
