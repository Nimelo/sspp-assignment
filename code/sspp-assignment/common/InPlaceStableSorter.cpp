#include "InPlaceStableSorter.h"

template<typename T>
inline void tools::sorters::InPlaceStableSorter::swap(T & lhs, T & rhs)
{
	T tmp = lhs;
	lhs = rhs;
	rhs = tmp;
}

void tools::sorters::InPlaceStableSorter::quicksort(int * I, int * J, FLOATING_TYPE * values, const int left, const int right)
{
	int i = left, j = right;
	int pivot = (I)[(left + right) / 2];

	while (i <= j) {
		while (I[i] < pivot)
			i++;
		while (I[j] > pivot)
			j--;
		if (i <= j) {
			if (I[i] != I[j]) {
				swap(I[i], I[j]);
				swap(J[i], J[j]);
				swap(values[i], values[j]);
			}	
			i++;
			j--;
		}
	};

	/* recursion */
	if (left < j)
		quicksort(I, J, values, left, j);
	if (i < right)
		quicksort(I, J, values, i, right);
}

void tools::sorters::InPlaceStableSorter::sort(int * I, int * J, FLOATING_TYPE * values, int N)
{
	quicksort(I, J, values, 0, N - 1);
	int beginIndex = 0;
	for (int i = 1; i < N; i++)
	{
		if (I[i - 1] != J[i])
		{
			quicksort(J, I, values, beginIndex, i - 1);
			beginIndex = i;
		}
	}
}