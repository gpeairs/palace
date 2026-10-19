// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <random>
#include <string>
#include <mfem.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "fem/errorindicator.hpp"
#include "fem/fespace.hpp"
#include "fem/integrator.hpp"
#include "fem/mesh.hpp"
#include "linalg/errorestimator.hpp"
#include "linalg/vector.hpp"
#include "models/materialoperator.hpp"
#include "utils/communication.hpp"
#include "utils/configfile.hpp"

namespace palace
{

using namespace Catch::Matchers;

namespace
{

auto LoadMesh(MPI_Comm comm, const std::string &input)
{
  mfem::Mesh smesh(input, 1, 1);
  smesh.EnsureNodes();
  for (int i = 0; i < smesh.GetNE(); i++)
  {
    smesh.SetAttribute(i, 1);
  }
  smesh.SetAttributes();
  REQUIRE(Mpi::Size(comm) <= smesh.GetNE());
  return Mesh(std::make_unique<mfem::ParMesh>(comm, smesh));
}

void FillRandom(Vector &v, unsigned seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  auto *h = v.HostWrite();
  for (int i = 0; i < v.Size(); i++)
  {
    h[i] = dist(gen);
  }
}

}  // namespace

// The error estimators integrate the flux error with one libCEED sub-operator per element
// geometry type, all reading the same field vectors. libCEED re-gathers a passive input
// only when its vector's state changes, and for complex fields the vectors are re-pointed
// from the real to the imaginary part between the two integration passes, so every
// sub-operator must see those updates: on a mixed mesh, a second estimate and the imaginary
// pass must not reuse the data of the first estimate for the geometry types other than the
// first.
TEST_CASE("Flux error estimators on mixed-geometry meshes",
          "[errorestimator][Serial][Parallel][GPU]")
{
  const auto comm = MPI_COMM_WORLD;
  auto mesh =
      LoadMesh(comm, std::string(PALACE_TEST_DATA_DIR "/mesh/fichera-mixed-p2.mesh"));
  const int dim = mesh.Dimension();
  REQUIRE(dim == 3);
  constexpr int order = 2;
  fem::DefaultIntegrationOrder::p_trial = order;

  config::MaterialData material;
  material.attributes = {1};
  material.epsilon_r.s = {2.0, 3.0, 4.0};
  material.mu_r.s = {1.0, 1.5, 2.0};
  config::PeriodicBoundaryData periodic;
  MaterialOperator mat_op({material}, periodic, ProblemType::EIGENMODE, mesh);

  mfem::ND_FECollection nd_fec(order, dim);
  mfem::RT_FECollection rt_fec(order - 1, dim);
  FiniteElementSpaceHierarchy nd_fespaces(
      std::make_unique<FiniteElementSpace>(mesh, &nd_fec));
  FiniteElementSpaceHierarchy rt_fespaces(
      std::make_unique<FiniteElementSpace>(mesh, &rt_fec));
  auto &nd_fespace = nd_fespaces.GetFinestFESpace();
  auto &rt_fespace = rt_fespaces.GetFinestFESpace();

  // Tight flux projection so that the complex (paired) and real solves agree well beyond
  // the tolerance used below.
  constexpr double tol = 1.0e-14;
  constexpr int max_it = 1000, print = 0;
  constexpr bool use_mg = false;

  auto Compare = [&](const Vector &actual, const Vector &expected, const char *what)
  {
    REQUIRE(actual.Size() == mesh.GetNE());
    REQUIRE(expected.Size() == actual.Size());
    const auto *ha = actual.HostRead();
    const auto *he = expected.HostRead();
    int num_bad = 0;
    double max_rel = 0.0;
    for (int i = 0; i < actual.Size(); i++)
    {
      const double rel = std::abs(ha[i] - he[i]) / std::max(std::abs(he[i]), 1.0e-300);
      max_rel = std::max(max_rel, rel);
      num_bad += (rel > 1.0e-8);
    }
    INFO(what << ": max relative difference " << max_rel);
    CHECK(num_bad == 0);
  };

  // MakeEstimator() constructs a fresh estimator of the requested type; a fresh estimator's
  // first estimate is always correct and serves as the reference.
  auto Run = [&](auto MakeEstimator, const FiniteElementSpace &fespace)
  {
    ComplexVector X(fespace.GetTrueVSize());
    X.UseDevice(true);
    FillRandom(X.Real(), 7);
    FillRandom(X.Imag(), 11);

    ErrorIndicator ref_re, ref_im;
    MakeEstimator(Vector()).AddErrorIndicator(X.Real(), 1.0, ref_re);
    MakeEstimator(Vector()).AddErrorIndicator(X.Imag(), 1.0, ref_im);
    {
      // The two parts give different estimates, so that reading the wrong part is detected.
      const auto *hre = ref_re.Local().HostRead();
      const auto *him = ref_im.Local().HostRead();
      int distinct = 0;
      for (int i = 0; i < ref_re.Local().Size(); i++)
      {
        distinct += (std::abs(hre[i] - him[i]) > 1.0e-6 * std::max(hre[i], him[i]));
      }
      Mpi::GlobalSum(1, &distinct, comm);
      REQUIRE(distinct > 0);
    }

    SECTION("Repeated real estimates")
    {
      auto estimator = MakeEstimator(Vector());
      ErrorIndicator first, second;
      estimator.AddErrorIndicator(X.Real(), 1.0, first);
      estimator.AddErrorIndicator(X.Imag(), 1.0, second);
      Compare(first.Local(), ref_re.Local(), "first real estimate");
      Compare(second.Local(), ref_im.Local(), "second real estimate");
    }

    SECTION("Complex estimate")
    {
      auto estimator = MakeEstimator(ComplexVector());
      ErrorIndicator ind_c;
      estimator.AddErrorIndicator(X, 1.0, ind_c);
      Vector expected(ref_re.Local().Size());
      {
        const auto *hre = ref_re.Local().HostRead();
        const auto *him = ref_im.Local().HostRead();
        auto *he = expected.HostWrite();
        for (int i = 0; i < expected.Size(); i++)
        {
          he[i] = std::sqrt(hre[i] * hre[i] + him[i] * him[i]);
        }
      }
      Compare(ind_c.Local(), expected, "complex estimate");
    }
  };

  SECTION("Curl flux estimator")
  {
    Run(
        [&](auto vec_type)
        {
          return CurlFluxErrorEstimator<decltype(vec_type)>(mat_op, rt_fespace, nd_fespaces,
                                                            tol, max_it, print, use_mg);
        },
        rt_fespace);
  }

  SECTION("Gradient flux estimator")
  {
    Run(
        [&](auto vec_type)
        {
          return GradFluxErrorEstimator<decltype(vec_type)>(mat_op, nd_fespace, rt_fespaces,
                                                            tol, max_it, print, use_mg);
        },
        nd_fespace);
  }

  SECTION("Time-dependent flux estimator field parts")
  {
    // The electric and magnetic field parts are the gradient and curl flux estimates with
    // the shared energy normalization, and combine to the total in quadrature.
    Vector E(nd_fespace.GetTrueVSize()), B(rt_fespace.GetTrueVSize());
    E.UseDevice(true);
    B.UseDevice(true);
    FillRandom(E, 7);
    FillRandom(B, 11);
    constexpr double Et = 2.5;

    ErrorIndicator ref_E, ref_B;
    GradFluxErrorEstimator<Vector>(mat_op, nd_fespace, rt_fespaces, tol, max_it, print,
                                   use_mg)
        .AddErrorIndicator(E, Et, ref_E);
    CurlFluxErrorEstimator<Vector>(mat_op, rt_fespace, nd_fespaces, tol, max_it, print,
                                   use_mg)
        .AddErrorIndicator(B, Et, ref_B);
    REQUIRE_FALSE(ref_E.GetSummaryStatistics(comm).eta_e);

    ErrorIndicator ind;
    TimeDependentFluxErrorEstimator<Vector>(mat_op, nd_fespaces, rt_fespaces, tol, max_it,
                                            print, use_mg)
        .AddErrorIndicator(E, B, Et, ind);

    Vector expected(ind.Local().Size());
    {
      const auto *he = ref_E.Local().HostRead();
      const auto *hb = ref_B.Local().HostRead();
      auto *hx = expected.HostWrite();
      for (int i = 0; i < expected.Size(); i++)
      {
        hx[i] = std::sqrt(he[i] * he[i] + hb[i] * hb[i]);
      }
    }
    Compare(ind.Local(), expected, "total from field parts");

    const auto stats = ind.GetSummaryStatistics(comm);
    REQUIRE(stats.eta_e);
    REQUIRE(stats.eta_m);
    CHECK_THAT(*stats.eta_e, WithinRel(ref_E.Norml2(comm), 1.0e-12));
    CHECK_THAT(*stats.eta_m, WithinRel(ref_B.Norml2(comm), 1.0e-12));
  }
}

// The per-sample global indicators and the field parts are kept alongside the running
// average, which must remain consistent with them.
TEST_CASE("ErrorIndicator field parts and sample norms",
          "[errorestimator][Serial][Parallel]")
{
  const auto comm = MPI_COMM_WORLD;
  constexpr int ne = 5;
  const int rank = Mpi::Rank(comm);
  auto Make = [&](double scale)
  {
    Vector v(ne);
    auto *h = v.HostWrite();
    for (int i = 0; i < ne; i++)
    {
      h[i] = scale * (1.0 + i + 0.5 * rank);
    }
    return v;
  };
  auto Total = [&](const Vector &E, const Vector &B)
  {
    Vector v(ne);
    const auto *he = E.HostRead();
    const auto *hb = B.HostRead();
    auto *h = v.HostWrite();
    for (int i = 0; i < ne; i++)
    {
      h[i] = std::sqrt(he[i] * he[i] + hb[i] * hb[i]);
    }
    return v;
  };

  // Two samples with different balances of electric and magnetic error.
  const Vector E1 = Make(1.0), B1 = Make(3.0), E2 = Make(4.0), B2 = Make(0.5);
  const Vector T1 = Total(E1, B1), T2 = Total(E2, B2);
  ErrorIndicator ind;
  ind.AddIndicator(T1, E1, B1);
  ind.AddIndicator(T2, E2, B2);

  const auto norms = ind.GetSampleNorms(comm);
  REQUIRE(norms.size() == 2);
  const double n1 = linalg::Norml2(comm, T1), n2 = linalg::Norml2(comm, T2);
  CHECK_THAT(norms[0].norm, WithinRel(n1, 1.0e-14));
  CHECK_THAT(norms[1].norm, WithinRel(n2, 1.0e-14));
  CHECK_THAT(*norms[0].eta_e, WithinRel(linalg::Norml2(comm, E1), 1.0e-14));
  CHECK_THAT(*norms[0].eta_m, WithinRel(linalg::Norml2(comm, B1), 1.0e-14));
  CHECK_THAT(*norms[1].eta_e, WithinRel(linalg::Norml2(comm, E2), 1.0e-14));
  CHECK_THAT(*norms[1].eta_m, WithinRel(linalg::Norml2(comm, B2), 1.0e-14));

  // The global indicator of the average and its field parts are the RMS of the per-sample
  // indicators, and the field parts combine to the total in quadrature.
  const auto stats = ind.GetSummaryStatistics(comm);
  CHECK_THAT(stats.norm * stats.norm, WithinRel(0.5 * (n1 * n1 + n2 * n2), 1.0e-14));
  CHECK_THAT(*stats.eta_e * *stats.eta_e,
             WithinRel(0.5 * (std::pow(*norms[0].eta_e, 2) + std::pow(*norms[1].eta_e, 2)),
                       1.0e-14));
  CHECK_THAT(*stats.eta_m * *stats.eta_m,
             WithinRel(0.5 * (std::pow(*norms[0].eta_m, 2) + std::pow(*norms[1].eta_m, 2)),
                       1.0e-14));
  CHECK_THAT(stats.norm * stats.norm,
             WithinRel(*stats.eta_e * *stats.eta_e + *stats.eta_m * *stats.eta_m, 1.0e-14));

  // Single-field indicators report per-sample norms without field parts.
  ErrorIndicator single;
  single.AddIndicator(E1);
  single.AddIndicator(E2);
  const auto single_norms = single.GetSampleNorms(comm);
  REQUIRE(single_norms.size() == 2);
  CHECK_FALSE(single_norms[0].eta_e);
  CHECK_THAT(single_norms[1].norm, WithinRel(linalg::Norml2(comm, E2), 1.0e-14));
  CHECK_FALSE(single.GetSummaryStatistics(comm).eta_e);
}

}  // namespace palace
