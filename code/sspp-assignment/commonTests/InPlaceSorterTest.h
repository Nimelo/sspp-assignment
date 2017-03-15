#pragma once
#include "ITest.h"
#include "InPlaceStableSorter.h"
class InPlaceSorterTest : public ITest
{
	protected:
		sspp::tools::sorters::InPlaceStableSorter *sorter;
		virtual void SetUp()
		{
			sorter = new sspp::tools::sorters::InPlaceStableSorter();
		}

		virtual void TearDown()
		{
			delete sorter;
		}
};