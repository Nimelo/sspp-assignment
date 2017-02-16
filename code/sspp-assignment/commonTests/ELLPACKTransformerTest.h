#pragma once

#include "ITest.h"
#include "ELLPACKTransformer.h"

class ELLPACKTransformerTest : public ITest
{
protected:
	tools::transformers::ellpack::ELLPACKTransformer *ellpackTransformer;
	virtual void SetUp()
	{
		ellpackTransformer = new tools::transformers::ellpack::ELLPACKTransformer();
	}

	virtual void TearDown()
	{
		delete ellpackTransformer;
	}
};
