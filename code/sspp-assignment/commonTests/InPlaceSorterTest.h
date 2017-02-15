#pragma once
#include "ITest.h"
#include "InPlaceSorter.h"
class InPlaceSorterTest : public ITest
{
	protected:
		tools::sorters::InPlaceSorter *sorter;
		virtual void SetUp()
		{
			sorter = new tools::sorters::InPlaceSorter();
		}

		virtual void TearDown()
		{
			delete sorter;
		}
};