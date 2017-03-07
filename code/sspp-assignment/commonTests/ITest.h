#pragma once

#include <iostream>
#include <gtest/gtest.h>
class ITest : public ::testing::Test
{
public:
	template<typename T>
	void assertArrays(T original, T current, int n, const char * str)
	{
		for (int i = 0; i < n; i++)
		{
			ASSERT_EQ(original[i], current[i]) << str + i;
		}
	}
};