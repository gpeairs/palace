// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "errorindicator.hpp"

#include <cmath>
#include <mfem/general/forall.hpp>

namespace palace
{

namespace
{

// Fold the indicator for one more sample into the running average over n samples.
void AddToAverage(Vector &local, const Vector &indicator, int n)
{
  if (n == 0)
  {
    local = indicator;
    return;
  }

  // The average local indicator is used rather than the indicator for the maximum
  // error to drive the adaptation, to account for a local error that might be marginally
  // important to many solves, rather than only large in one solve.
  MFEM_ASSERT(local.Size() == indicator.Size(),
              "Unexpected size mismatch for ErrorIndicator::AddIndicator!");

  // The local indicators must be squared before combining, so that the global error
  // calculation is valid:
  //                            E = √(1/N ∑ₙ ∑ₖ ηₖₙ²)
  // from which it follows that:
  //                            E² = 1/N ∑ₙ ∑ₖ ηₖₙ²
  //                               = 1/N ∑ₙ Eₙ²
  // Namely the average of the global error indicators included in the reduction.
  // Squaring both sides means the summation can be rearranged, and then the local error
  // indicators become:
  //                            eₖ = √(1/N ∑ₙ ηₖₙ²)
  const bool use_dev = local.UseDevice() || indicator.UseDevice();
  const int N = local.Size();
  const int Dn = n;
  const auto *DI = indicator.Read();
  auto *DL = local.ReadWrite();
  mfem::forall_switch(
      use_dev, N, [=] MFEM_HOST_DEVICE(int i)
      { DL[i] = std::sqrt((DL[i] * DL[i] * Dn + DI[i] * DI[i]) / (Dn + 1)); });
}

}  // namespace

void ErrorIndicator::AddIndicator(const Vector &indicator)
{
  MFEM_VERIFY(sample_sq_E.empty(),
              "ErrorIndicator samples must either all include or all omit the electric and "
              "magnetic field parts!");
  AddToAverage(local, indicator, n);
  sample_sq.push_back(linalg::LocalDot(indicator, indicator));

  // More samples have been added, update for the running average.
  n += 1;
}

void ErrorIndicator::AddIndicator(const Vector &indicator, const Vector &indicator_E,
                                  const Vector &indicator_B)
{
  MFEM_VERIFY(sample_sq_E.size() == sample_sq.size(),
              "ErrorIndicator samples must either all include or all omit the electric and "
              "magnetic field parts!");
  MFEM_ASSERT(indicator_E.Size() == indicator.Size() &&
                  indicator_B.Size() == indicator.Size(),
              "Unexpected size mismatch for ErrorIndicator::AddIndicator!");
  AddToAverage(local, indicator, n);
  sample_sq.push_back(linalg::LocalDot(indicator, indicator));
  sample_sq_E.push_back(linalg::LocalDot(indicator_E, indicator_E));
  sample_sq_B.push_back(linalg::LocalDot(indicator_B, indicator_B));

  // More samples have been added, update for the running average.
  n += 1;
}

ErrorIndicator::SummaryStatistics ErrorIndicator::GetSummaryStatistics(MPI_Comm comm) const
{
  SummaryStatistics stats{Norml2(comm), Min(comm), Max(comm), Mean(comm)};
  if (!sample_sq_E.empty())
  {
    // The running average is linear in the squared elemental indicators, so the squared
    // global indicator of each field part of the average is the mean over samples.
    double sq[2] = {0.0, 0.0};
    for (std::size_t i = 0; i < sample_sq_E.size(); i++)
    {
      sq[0] += sample_sq_E[i];
      sq[1] += sample_sq_B[i];
    }
    Mpi::GlobalSum(2, sq, comm);
    stats.eta_e = std::sqrt(sq[0] / n);
    stats.eta_m = std::sqrt(sq[1] / n);
  }
  return stats;
}

std::vector<ErrorIndicator::SampleNorms> ErrorIndicator::GetSampleNorms(MPI_Comm comm) const
{
  // Reduce the sums for all samples and field parts at once.
  const std::size_t ns = sample_sq.size();
  const bool has_field_parts = !sample_sq_E.empty();
  std::vector<double> sq(sample_sq);
  if (has_field_parts)
  {
    sq.insert(sq.end(), sample_sq_E.begin(), sample_sq_E.end());
    sq.insert(sq.end(), sample_sq_B.begin(), sample_sq_B.end());
  }
  Mpi::GlobalSum(static_cast<int>(sq.size()), sq.data(), comm);

  std::vector<SampleNorms> norms(ns);
  for (std::size_t i = 0; i < ns; i++)
  {
    norms[i].norm = std::sqrt(sq[i]);
    if (has_field_parts)
    {
      norms[i].eta_e = std::sqrt(sq[ns + i]);
      norms[i].eta_m = std::sqrt(sq[2 * ns + i]);
    }
  }
  return norms;
}

}  // namespace palace
