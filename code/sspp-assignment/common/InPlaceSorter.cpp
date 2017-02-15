#include "InPlaceSorter.h"

template<typename T>
inline void tools::sorters::InPlaceSorter::swap(T & lhs, T & rhs)
{
	T tmp = lhs;
	lhs = rhs;
	rhs = tmp;
}

template<typename T>
void tools::sorters::InPlaceSorter::quicksort(int * I, int * J, FLOATING_TYPE * values, const T left, const T right)
{
	int i = left, j = right;
	int pivot = (I)[(left + right) / 2];

	while (i <= j) {
		while (I[i] < pivot)
			i++;
		while (I[j] > pivot)
			j--;
		if (i <= j) {
			swap(I[i], I[j]);
			swap(J[i], J[j]);
			swap(values[i], values[j]);
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

void tools::sorters::InPlaceSorter::sort(int * I, int * J, FLOATING_TYPE * values, int N)
{
	quicksort(I, J, values, 0, N - 1);
}