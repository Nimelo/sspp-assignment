#include "ELLPACKTransformer.h"
#include "InPlaceStableSorter.h"
#include <algorithm>

representations::ellpack::ELLPACK tools::transformers::ellpack::ELLPACKTransformer::transform(const representations::intermediary::IntermediarySparseMatrix & ism)
{
	tools::sorters::InPlaceStableSorter sorter;
	sorter.sort(ism.IIndexes, ism.JIndexes, ism.Values, ism.NZ);
	
	int N = ism.N, M = ism.M, NZ = ism.NZ, MAXNZ;
	int **JA;
	FLOATING_TYPE **AS;

	int * auxArray = new int[M];

	int tmp = 0;
	int index = 0;
	for (int i = 1; i < NZ; i++)
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

	if (tmp != 0)
		auxArray[index++] = ++tmp;
	else
		auxArray[index++] = 1;

	MAXNZ = *std::max_element(auxArray, auxArray + NZ - 1);

	JA = new int*[M];
	for (int i = 0; i < M; i++)
		JA[i] = new int[MAXNZ];

	AS = new FLOATING_TYPE*[M];
	for (int i = 0; i < M; i++)
		AS[i] = new FLOATING_TYPE[MAXNZ];

	int nzIndex = 0;

	for (int row = 0; row < M; row++)
	{
		for (int column = 0; column < auxArray[row]; column++)
		{
			AS[row][column] = ism.Values[nzIndex];
			JA[row][column] = ism.JIndexes[nzIndex];
			++nzIndex;
		}
			
		if (auxArray[row] < MAXNZ)
		{
			for (int i = auxArray[row]; i < MAXNZ; i++)
			{
				AS[row][i] = 0;
				if (auxArray[row] != 0)
					JA[row][i] = JA[row][i - 1];
				else
					JA[row][i] = 0;
			}
		}
	}

	delete[] auxArray;
	return representations::ellpack::ELLPACK(M, N, NZ, MAXNZ, JA, AS);
}
