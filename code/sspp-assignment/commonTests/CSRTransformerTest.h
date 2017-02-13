#pragma once

#include "ITest.h"
#include "CSRTransformer.h"

class CSRTransformerTest : public ITest
{
	protected:
		tools::transformers::csr::CSRTransformer *csrTransformer;
		virtual void SetUp()
		{
			csrTransformer = new tools::transformers::csr::CSRTransformer();
		}

		virtual void TearDown()
		{
			delete csrTransformer;
		}
};