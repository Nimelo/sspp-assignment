#pragma once
#include "ITest.h"
#include "InPlaceStableSorter.h"
class InPlaceSorterTest : public ITest
{
	protected:
		tools::sorters::InPlaceStableSorter *sorter;
		virtual void SetUp()
		{
			sorter = new tools::sorters::InPlaceStableSorter();
		}

		virtual void TearDown()
		{
			delete sorter;
		}
};