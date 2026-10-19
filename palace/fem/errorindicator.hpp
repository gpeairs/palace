// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef PALACE_FEM_ERROR_INDICATORS_HPP
#define PALACE_FEM_ERROR_INDICATORS_HPP

#include <array>
#include <optional>
#include <vector>
#include "linalg/vector.hpp"
#include "utils/communication.hpp"

namespace palace
{

//
// Storage for error estimation results from a simulation which involves one or more solves,
// required in the AMR loop.
//
class ErrorIndicator
{
protected:
  // Elemental localized error indicators. Used for marking elements for
  // refinement and coarsening.
  Vector local;

  // Rank-local sums over elements of the squared indicator of each sample, in the order the
  // samples were added, and of its electric and magnetic field parts when every sample was
  // added with them. The running average in local discards the global indicator of the
  // individual samples, which these preserve.
  std::vector<double> sample_sq, sample_sq_E, sample_sq_B;

  // Number of samples.
  int n;

public:
  ErrorIndicator(Vector &&local)
    : local(std::move(local)), sample_sq{linalg::LocalDot(this->local, this->local)}, n(1)
  {
    this->local.UseDevice(true);
  }
  ErrorIndicator() : n(0) { local.UseDevice(true); }

  // Add an indicator to the running total.
  void AddIndicator(const Vector &indicator);

  // Add an indicator to the running total along with its electric and magnetic field parts,
  // where indicator² = indicator_E² + indicator_B² for each element.
  void AddIndicator(const Vector &indicator, const Vector &indicator_E,
                    const Vector &indicator_B);

  // Return the local error indicator.
  const auto &Local() const { return local; }

  // Return the global error indicator.
  auto Norml2(MPI_Comm comm) const { return linalg::Norml2(comm, local); }

  // Return the largest local error indicator.
  auto Max(MPI_Comm comm) const
  {
    auto max = local.Max();
    Mpi::GlobalMax(1, &max, comm);
    return max;
  }

  // Return the smallest local error indicator.
  auto Min(MPI_Comm comm) const
  {
    auto min = local.Min();
    Mpi::GlobalMin(1, &min, comm);
    return min;
  }

  // Return the mean local error indicator.
  auto Mean(MPI_Comm comm) const { return linalg::Mean(comm, local); }

  struct SummaryStatistics
  {
    double norm;
    double min;
    double max;
    double mean;

    // Global indicators of the electric and magnetic field parts, norm² = η_e² + η_m², when
    // available.
    std::optional<double> eta_e, eta_m;
  };

  SummaryStatistics GetSummaryStatistics(MPI_Comm comm) const;

  // Global indicator of each sample in the order added, with the electric and magnetic
  // field parts when available. The global indicator of the running average is the
  // root-mean-square of these: Norml2² = 1/N ∑ₙ normₙ².
  struct SampleNorms
  {
    double norm;
    std::optional<double> eta_e, eta_m;
  };

  std::vector<SampleNorms> GetSampleNorms(MPI_Comm comm) const;
};

}  // namespace palace

#endif  // PALACE_FEM_ERROR_INDICATORS_HPP
