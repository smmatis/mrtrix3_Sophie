/* Copyright (c) 2008-2024 the MRtrix3 contributors.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Covered Software is provided under this License on an "as is"
 * basis, without warranty of any kind, either expressed, implied, or
 * statutory, including, without limitation, warranties that the
 * Covered Software is free of defects, merchantable, fit for a
 * particular purpose or non-infringing.
 * See the Mozilla Public License v. 2.0 for more details.
 *
 * For more details, see http://www.mrtrix.org/.
 */

#include <limits>
#include <string>
#include <vector>

#include "command.h"
#include "header.h"
#include "image.h"
#include "image_helpers.h"
#include "phase_encoding.h"
#include "progressbar.h"
#include "adapter/gradient1D.h"
#include "dwi/gradient.h"
#include "dwi/shells.h"
#include "math/sphere.h"
#include "math/SH.h"

using namespace MR;
using namespace App;

const char *const operations[] = {"combine_pairs",
                                  "leave_one_out",
                                  "combine_predicted",
                                  nullptr};

// clang-format off
void usage() {

  AUTHOR = "Robert E. Smith (fobert.smith@florey.edu.au)";

  SYNOPSIS = "Perform reconstruction of DWI data from an input DWI series";

  DESCRIPTION
  + "This command provides a range of mechanisms by which to reconstruct estimated DWI data"
    " given a set of input DWI data and possibly other information about the image"
    " acquisition and/or reconstruction process. "
    "The operation that is appropriate for a given workflow is entirely dependent"
    " on the context of the details of that workflow and how the image data were acquired. "
    "Each operation available is described in further detail below."

  + "The \"combine_pairs\" operation is applicable in the scenario where the DWI acquisition"
    " involves acquiring the same diffusion gradient table twice,"
    " with the direction of phase encoding reversed in the second acquisition. "
    "It is a requirement in this case that the total readout time be equivalent between the two series;"
    " that is, they vary based only on the direction of phase encoding, not the speed."
    "The purpose of this command in that context is to take as input the full set of volumes"
    " (ie. from both phase encoding directions),"
    " find those pairs of DWI volumes with equivalent diffusion sensitisation"
    " but opposite phase encoding direction,"
    " and explicitly combine each pair into a single output volume,"
    " where the contribution of each image in the pair to the output image intensity"
    " is modulated by the relative Jacobians of the two distorted images."

  + "The \"leave_one_out\" operation derives an estimate for the DWI signal intensity"
    " for each sample based on all other samples in that voxel,"
    " and generates the output image based on all such estimates."
    " NOTE: NOT YET IMPLEMENTED"

  + "The \"combine_predicted\" operation is intended for DWI acquisition designs"
    " where the diffusion gradient table is split between different phase encoding directions."
    " Here, where there is greater uncertainty in what the DWI signal should look like"
    " due to susceptibility-driven signal compression in the acquired image data,"
    " the reconstructed image will be moreso influenced by the signal intensity"
    " that is estimated from those volumes with different phase encoding"
    " that did not experience such compression."
    " This is intended to act as a surrogate for weighted model fitting"
    " where the downstream model is not yet compatible with taking user-specified"
    " weights into account."
    " NOTE: NOT YET IMPLEMENTED";

  ARGUMENTS
    + Argument ("input", "the input DWI series").type_image_in()
    + Argument ("operation", "the way in which output DWIs will be reconstructed;"
                " one of: " + join(operations, ", ")).type_choice(operations)
    + Argument ("output", "the output DWI series").type_image_out();

  OPTIONS
    + Option("field", "provide a B0 field offset image in Hz")
      + Argument("image").type_image_in()

    + Option("lmax", "set the maximal spherical harmonic degrees to use (one for each b-value) during signal reconstruction")
      + Argument("value").type_sequence_int()

    // TODO Appropriate to have other command-line options to specify the phase encoding design?
    + PhaseEncoding::ImportOptions
    + PhaseEncoding::ExportOptions
    + DWI::GradImportOptions()
    + DWI::GradExportOptions();

}
// clang-format on


using scheme_type = Eigen::Matrix<default_type, Eigen::Dynamic, Eigen::Dynamic>;
using spherical_scheme_type = Eigen::Matrix<default_type, Eigen::Dynamic, 2>;
using sh_transform_type = Eigen::Matrix<default_type, Eigen::Dynamic, Eigen::Dynamic>;
using data_vector_type = Eigen::Matrix<default_type, Eigen::Dynamic, 1>;


//////////////////////
// Shared functions //
//////////////////////

Image<float> get_field_image(const Image<float> dwi_in,
                             const std::string& operation,
                             const bool compulsory) {

  auto opt = get_options("field");
  Image<float> field_image;
  if (opt.empty()) {
    if (compulsory)
      throw Exception("-field option is compulsory for \"" + operation + "\" operation");
    WARN("No susceptibility field image provided for \"" + operation + "\" operation; "
         "some functionality will be omitted");
  } else {
    field_image = Image<float>::open(std::string(opt[0][0]));
    if (!voxel_grids_match_in_scanner_space(dwi_in, field_image))
      throw Exception("Susceptibility field image and DWI series not defined on same voxel grid");
    if (!(field_image.ndim() == 3 || (field_image.ndim() == 4 && field_image.size(3) == 1)))
      throw Exception ("Susceptibility field image expected to be 3D");
  }
  return field_image;
}



// Generate mapping from volume index to shell index
std::vector<int> get_vol2shell(const DWI::Shells &shells, const size_t volume_count) {
  std::vector<int> vol2shell(volume_count, -1);
  for (size_t shell_index = 0; shell_index != shells.count(); ++shell_index) {
    const DWI::Shell &shell(shells[shell_index]);
    for (const auto &volume_index : shell.get_volumes()) {
      assert (vol2shell[volume_index] == -1);
      vol2shell[volume_index] = shell_index;
    }
  }
  assert (std::min(vol2shell.begin(), vol2shell.end()) == 0);
  return vol2shell;
}



// Find:
//   - Which image axis the gradient is to be computed along
//   - Whether the sign needs to be negated
std::pair<size_t, default_type> get_pe_axis_and_sign(const Eigen::Block<scheme_type, 1, 3> pe_dir) {
  for (size_t axis_index = 0; axis_index != 3; ++axis_index) {
    if (pe_dir[axis_index] != 0)
      return std::make_pair(axis_index, pe_dir[axis_index] > 0 ? 1.0 : -1.0);
  }
  assert (false);
  return std::make_pair(size_t(3), std::numeric_limits<default_type>::signaling_NaN());
}





/////////////////////////////////////////
// Functions for individual operations //
/////////////////////////////////////////

void run_combine_pairs(Image<float> &dwi_in,
                       const scheme_type &grad_in,
                       const scheme_type &pe_in,
                       Header &header_out) {

  if (grad_in.rows() % 2)
    throw Exception("Cannot perform explicit volume recombination based on phase encoding pairs:"
                    " number of volumes is odd");

  const std::vector<std::string> invalid_options {"lmax"};
  for (const auto opt : invalid_options)
    if (!get_options(opt).empty())
      throw Exception("-" + opt + " option not supported for \"combine_pairs\" operation");

  Image<float> field_image = get_field_image(dwi_in, "combine_pairs", false);

  scheme_type pe_config;
  Eigen::Array<int, Eigen::Dynamic, 1> pe_indices;
  PhaseEncoding::scheme2eddy(pe_in, pe_config, pe_indices);
  if (pe_config.rows() % 2)
    throw Exception("Cannot perform explicit volume recombination based on phase encoding pairs:"
                    " number of unique phase encodings is odd");
  // The FSL topup / eddy format indexes from one;
  //   change to starting from zero for internal array indexing
  pe_indices -= 1;

  // Ensure that for each line in pe_config,
  //   there is a corresponding line with the same total readout time
  //   but opposite phase encoding
  std::vector<std::pair<size_t, size_t>> pe_pairs;
  std::vector<int> peindex2paired(pe_config.rows(), -1);
  for (size_t pe_first_index = 0; pe_first_index != pe_config.rows(); ++pe_first_index) {
    if (peindex2paired[pe_first_index] >= 0)
      continue;
    const auto pe_first = pe_config.row(pe_first_index);
    size_t pe_second_index;
    for (pe_second_index = pe_first_index + 1; pe_second_index != pe_config.rows(); ++pe_second_index) {
      const auto pe_second = pe_config.row(pe_second_index);
      if (((pe_second.head(3) + pe_first.head(3)).squaredNorm() == 0) && // Phase encoding same axis but reversed direction
          (pe_second[3] && pe_first[3])) {                               // Equal total readout time
        peindex2paired[pe_first_index] = pe_second_index;
        peindex2paired[pe_second_index] = pe_first_index;
        pe_pairs.push_back(std::make_pair(pe_first_index, pe_second_index));
        break;
      }
    }
    if (pe_second_index == pe_config.rows())
      throw Exception ("Unable to find corresponding reversed phase encoding volumes for:" //
                       " [" + str(pe_first.transpose()) + "]");
  }
  assert(std::min(peindex2paired.begin(), peindex2paired.end()) == 0);

  DWI::Shells shells(grad_in);
  const std::vector<int> vol2shell = get_vol2shell(shells, grad_in.rows());

  // Figure out for each volume in the output image
  //   which volumes in the input image will be contributing to its generation
  // As we do this,
  //   generate what the final diffusion gradient table is going to look like,
  //   since we'll need that to be pre-populated to initialise the output image
  //
  // TODO A potential enhancement here would be to improve matching
  //   in the scenario of considerable subject rotation between phase encoding directions
  // Just increasing the dot product threshold wouldn't be optimal in this case
  // Better would be to find, for each volume, the most suitable corresponding volume,
  //   and then make sure that there are no duplicates in that pairing
  std::vector<std::pair<size_t, size_t>> volume_pairs;
  volume_pairs.reserve(grad_in.rows() / 2);
  std::vector<int> in2outindex(grad_in.rows(), -1);
  scheme_type grad_out(scheme_type::Constant(grad_in.rows() / 2, 4, std::numeric_limits<default_type>::quiet_NaN()));
  size_t out_volume = 0;
  for (size_t first_volume = 0; first_volume != grad_in.rows(); ++first_volume) {
    // Volume is already assigned to a pair
    if (in2outindex[first_volume] >= 0)
      continue;
    // Which phase encoding group does this volume belong to?
    const size_t pe_first_index = pe_indices[first_volume];
    // Which phase encoding group must the paired volume therefore belong to?
    const size_t pe_second_index = peindex2paired[pe_first_index];
    // Is this a b=0 volume?
    const bool is_bzero = shells[vol2shell[first_volume]].is_bzero();
    // Gradient direction & b-value
    const auto first_dir = grad_in.block<1,3>(first_volume, 0);
    size_t second_volume;
    for (second_volume = first_volume + 1; second_volume != grad_in.rows(); ++second_volume) {
      // Can't match to this volume; it's already been paired off
      if (in2outindex[second_volume] >= 0)
        continue;
      // Doesn't belong to the right phase encoding group
      if (pe_indices[second_volume] != pe_second_index)
        continue;
      // Doesn't belong to the same shell
      if (vol2shell[second_volume] != vol2shell[first_volume])
        continue;
      const auto second_dir = grad_in.block<1,3>(second_volume, 0);
      // Only test for equivalence of gradient vectors if this isn't the b=0 shell
      if (!is_bzero) {
        // Some of the code below might be redundant
        //   given the prior checking of whether these volumes are ascribed to a b=0 shell;
        //   it's nevertheless duplicated from dwifslpreproc here
        if (first_dir.squaredNorm() > 0.0) {
          // One is zero, the other is not; therefore not a match
          if (second_dir.squaredNorm() == 0.0)
            continue;
          if (std::abs(first_dir.dot(second_dir)) < 0.999)
            continue;
        // One is zero, the other is not; therefore not a match
        } else if (second_dir.squaredNorm() > 0.0) {
          continue;
        }
      }
      // We now consider this pair of volumes to be a match
      Eigen::Vector3d average_dir = 0.5 * (first_dir + second_dir);
      // Directions may be of opposite polarity
      if (average_dir.squaredNorm() < 0.5)
        average_dir =  0.5 * (first_dir - second_dir);
      // Allow to remain as [0.0, 0.0, 0.0]
      if (average_dir.squaredNorm() > 0.0)
        average_dir.normalize();
      const default_type average_bvalue = 0.5 * (grad_in(first_volume, 3) + grad_in(second_volume, 3));
      grad_out.block(out_volume, 0, 1, 3) = average_dir;
      grad_out(out_volume, 3) = average_bvalue;
      volume_pairs.push_back(std::make_pair(first_volume, second_volume));
      in2outindex[first_volume] = in2outindex[second_volume] = out_volume;
      ++out_volume;
    }
    if (second_volume == grad_in.rows())
      throw Exception("Unable to establish paired DWI volume with reversed phase encoding:"
                      " index " + str(first_volume) + ";"
                      " grad " + str(grad_in.row(first_volume).transpose()) + ";"
                      " phase encoding " + str(pe_config.row(pe_first_index).transpose()));
  }
  assert (grad_out.allFinite());

  header_out.size(3) = dwi_in.size(3) / 2;
  DWI::set_DW_scheme(header_out, grad_out);
  Image<float> dwi_out = Image<float>::create(header_out.name(), header_out);

  if (field_image.valid()) {

    // TODO For now, going to compute and store both the jacobians and the weights;
    //   partly to be consistent with prior dwifslpreprpc code,
    //   partly because exporting these data might be of some utility
    // Could later remove explicit storage of Jacobians
    std::vector<Image<float>> jacobian_images;
    std::vector<Image<float>> weight_images;
    {
      // Need to calculate the "weight" to be applied to each phase encoding group during volume recombination
      // This is based on the Jacobian of the field along the phase encoding direction,
      //   scaled by the total readout time
      Adapter::Gradient1D<Image<float>> gradient(field_image);
      ProgressBar progress("Computing phase encoding group weighting images", pe_config.rows());
      for (size_t pe_index = 0; pe_index != pe_config.rows(); ++pe_index) {
        Image<float> jacobian_image = Image<float>::scratch(field_image, "Scratch Jacobian image for PE index " + str(pe_index));
        Image<float> weight_image = Image<float>::scratch(field_image, "Scratch weight image for PE index " + str(pe_index));
        const auto pe_axis_and_multiplier = get_pe_axis_and_sign(pe_config.block<1, 3>(pe_index, 0));
        gradient.set_axis(pe_axis_and_multiplier.first);
        const default_type multiplier = pe_axis_and_multiplier.second * pe_config(pe_index, 3);
        for (auto l = Loop(gradient) (gradient, jacobian_image, weight_image); l; ++l) {
          const default_type jacobian = std::max(0.0, 1.0 + (gradient.value() * multiplier));
          jacobian_image.value() = jacobian;
          weight_image.value() = Math::pow2(jacobian);
        }
        jacobian_images.push_back(std::move(jacobian_image));
        weight_images.push_back(std::move(weight_image));
        ++progress;
      }
    }
    ProgressBar progress("Performing explicit volume recombination", header_out.size(3));
    for (size_t out_volume = 0; out_volume != header_out.size(3); ++out_volume) {
      dwi_out.index(3) = out_volume;
      Image<float> first_volume(dwi_in), second_volume(dwi_in);
      first_volume.index(3) = volume_pairs[out_volume].first;
      second_volume.index(3) = volume_pairs[out_volume].second;
      Image<float> first_weight(weight_images[volume_pairs[out_volume].first]);
      Image<float> second_weight(weight_images[volume_pairs[out_volume].second]);
      for (auto l = Loop(dwi_out, 0, 3)(dwi_out, first_volume, second_volume, first_weight, second_weight); l; ++l) {
        dwi_out.value() = ((first_volume.value() * first_weight.value()) + (second_volume.value() * second_weight.value())) / //
                          (first_weight.value() + second_weight.value());
      }
      ++progress;
    }

  } else {

    // No field map image provided; do a straight averaging of input volumes into output
    ProgressBar progress("Performing explicit volume recombination", header_out.size(3));
    for (size_t out_volume = 0; out_volume != header_out.size(3); ++out_volume) {
      Image<float> first_volume(dwi_in), second_volume(dwi_in);
      dwi_out.index(3) = out_volume;
      first_volume.index(3) = volume_pairs[out_volume].first;
      second_volume.index(3) = volume_pairs[out_volume].second;
      for (auto l = Loop(dwi_out, 0, 3)(dwi_out, first_volume, second_volume); l; ++l)
        dwi_out.value() = 0.5 * (first_volume.value() + second_volume.value());
      ++progress;
    }

  }

}



// TODO Identify code from combine_pairs that can be shared
void run_combine_predicted(Image<float> &dwi_in,
                          const scheme_type &grad_in,
                          const scheme_type &pe_in,
                          Header &header_out) {

  Image<float> field_image = get_field_image(dwi_in, "combine_predicted", true);

  scheme_type pe_config;
  Eigen::Array<int, Eigen::Dynamic, 1> pe_indices;
  PhaseEncoding::scheme2eddy(pe_in, pe_config, pe_indices);
  // The FSL topup / eddy format indexes from one;
  //   change to starting from zero for internal array indexing
  pe_indices -= 1;

  DWI::Shells shells(grad_in);
  const std::vector<int> vol2shell = get_vol2shell(shells, grad_in.rows());

  auto opt = get_options("lmax");
  std::vector<int> lmax_user;
  if (!opt.empty()) {
    lmax_user = parse_ints<int>(opt[0][0]);
    if (lmax_user.size() != shells.count())
      throw Exception ("-lmax option must specify one lmax for each unique b-value");
    for (size_t shell_index = 0; shell_index != shells.count(); ++shell_index) {
      if (lmax_user[shell_index] % 2)
        throw Exception ("-lmax values must be even numbers");
      // TODO Technically this is a weak constraint:
      //   user-requested lmax may not be possible once excluding a phase encoding group
      if (lmax_user[shell_index] > Math::SH::NforL(shells[shell_index].count()))
        throw Exception ("Requested lmax=" + str(lmax_user[shell_index]) +
                         " for shell b=" + str<int>(shells[shell_index].get_mean()) +
                         ", but only " + str(shells[shell_index].count()) +
                         " volumes, which only supports lmax=" + str(Math::SH::NforL(shells[shell_index].count())));
    }
  }

  Adapter::Gradient1D<Image<float>> gradient(field_image);

  // TODO Perform check to ensure that within any phase encoding group,
  //   for each shell within that group,
  //   there is at least one volume present within at least one phase encoding block
  //   which therefore means that estimates can be generated in all circumstances

  // TODO Novel from here
  // - For each phase encoding group,
  //     generate a 4D image,
  //     where the value in each volume corresponds to the weight to be applied
  //     to the reconstructed intensity from that phase encoding group
  //   Here, I think I want to split the expression into two groups:
  //     1. The weight to attribute to the empirical signal itself
  //        vs. the sum of reconstructed estimates
  //     2. The relative weights to apply to predictions from the other phase encoding groups
  //   In both cases, derivation should be comparable to Skare 2010
  //
  // TODO Ideally, rather than each other phase encoding group contirbuting its own estimate,
  //   all other phase encoding groups would contribute in a weighted manner to a single estimate
  // This will require definition of a weighted SH fit,
  //   which was already a prospective addition as something to mimic properties of the GP predictor
  //
  // TODO There may be a better alternative expression for weighting empirical data vs. predictions
  // Currently, expression is:
  // - If Jacobian >= 1.0, use empirical data
  // - Otherwise, use (Jacobian * empirical) * ((1-Jacobian) * prediction)
  // This however fails to consider that the prediction could itself be of poor quality
  //   if it (hypothetically) also has a small Jacobian
  // This would only be relevant for fairly non-standard acquisitions,
  //   so might put this off for now

  // TODO Immediately generate Jacobian images for each phase encoding group;
  //   these can then be used for both:
  //   - Computing the weight to be attributed to the empirical data in output data generation
  //   - Construction of weighted SH fit
  std::vector<Image<float>> jacobian_images;
  for (size_t pe_index = 0; pe_index != pe_config.rows(); ++pe_index) {
    const auto pe_axis_and_multiplier = get_pe_axis_and_sign(pe_config.block<1, 3>(pe_index, 0));
    gradient.set_axis(pe_axis_and_multiplier.first);
    const default_type multiplier = pe_axis_and_multiplier.second * pe_config(pe_index, 3);
    Image<float> jacobian_image = Image<float>::scratch(field_image, "Jacobian image for phase encoding group " + str(pe_index));
    for (auto l = Loop(gradient) (gradient, jacobian_image); l; ++l)
      jacobian_image.value() = std::max(0.0, 1.0 + (gradient.value() * multiplier));
    jacobian_images.push_back(std::move(jacobian_image));
  }

  Image<float> dwi_out = Image<float>::create(header_out.name(), header_out);

  ProgressBar progress("Reconstructing volumes combining empirical and predicted intensities", pe_config.rows() * shells.count());
  for (size_t pe_index = 0; pe_index != pe_config.rows(); ++pe_index) {

    // For the empirical data within this phase encoding group,
    //   the jacobian is used directly as the weighted fraction by which
    //   the empirical input intensities will contribute to the output intensities
    // If the jacobian is 1.0 or greater,
    //   then the empirical data will be used as-is
    // If between 0.0 and 1.0,
    //   then (1.0 - value) will be the weighting fraction with which
    //   the predictions from other phase encoding groups will contribute
    // TODO Consider making this more preservative; eg. sqrt(jacobian)
    // TODO Also here for now we are assuming that from a single A2SH transformation (per shell),
    //   we can then do a single SH2A transformation to get all of the amplitudes of interest for this phase encoding group (per shell);
    //   in the future want to explore the prospect of additionally weighting by proximity to sample of interest,
    //   in which case there will be one A->SH->A transformation _per output volume_

    // TODO Branch depending on how many other phase encoding groups there are:
    //  - If only one other, then just construct the A->SH->A transform for that one group;
    //  - If more than one, construct the weighted A->SH->A transform considering all other groups
    // In both cases, the output amplitude directions are based on the phase encoding group currently being reconstructed,
    //   whereas the input amplitude directions are those of all other phase encoding groups
    // However a key difference is that:
    //   - In the situation where there's only one other phase encoding group,
    //     we can pre-compute the source2SH transformation
    //     (and therefore the source2target transformation)
    //     just once, and use it for every voxel,
    //     merely modulating how much the prediction vs. the empirical data contribute in each voxel
    //   - If using data from all other phase encoding groups together,
    //     the source2SH transformation has to be recomputed in every voxel
    //     (and will therefore likely be considerably slower)
    //
    // TODO Reconsider how predictions are generated:
    // Once a weighted fit is performed,
    //   the prediction could actually make use of information within the phase encoding group of interest,
    //   and this could nevertheless be integrated into generation of predictions
    // Among other things,
    //   this would remove the complications of checking whether a user-requested lmax is sensible,
    //   since it currently doesn't consider that SH transforms are computed while omitting data
    // It would however arguably place greater priority on:
    // - Use of leave-one-out
    // - Weighting samples by proximity

    // Loop over shells
    for (size_t shell_index = 0; shell_index != shells.count(); ++shell_index) {

      // Obtain volumes that belong both to this shell and:
      // - To the source phase encoding group; or
      // - To any other phase encoding group
      std::vector<size_t> source_volumes;
      std::vector<size_t> target_volumes;
      for (const auto volume : shells[shell_index].get_volumes()) {
        if (pe_indices[volume] == pe_index)
          target_volumes.push_back(volume);
        else
          source_volumes.push_back(volume);
      }
      assert(!source_volumes.empty());
      assert(!target_volumes.empty());
      const size_t lmax_data = Math::SH::LforN(source_volumes.size());
      size_t lmax;
      if (lmax_user.empty()) {
        lmax = lmax_data;
      } else {
        lmax = lmax_user[shell_index];
        if (lmax > lmax_data)
          throw Exception ("User-requested lmax=" + str(lmax) +
                           " for shell b=" + str<int>(shells[shell_index].get_mean()) +
                           " exceeds what can be predicted from data after phase encoding group exclusion");
      }

      // Generate the direction set for the target data
      spherical_scheme_type target_dirset(target_volumes.size(), 2);
      for (size_t target_index = 0; target_index != target_volumes.size(); ++target_index)
        Math::Sphere::cartesian2spherical(grad_in.block<1,3>(target_volumes[target_index], 0), target_dirset.row(target_index));
      // Generate the transformation from SH to the target data
      // TODO Need to confirm behaviour when the lmax of the source data exceeds
      //   what can actually be achieved for the target data in constructing the inverse transform
      const sh_transform_type SH2target = Math::SH::init_transform(target_dirset, lmax);

      // TODO Confirm that pre-allocated sizes are preserved
      spherical_scheme_type source_dirset(source_volumes.size(), 2);
      data_vector_type source_data(source_volumes.size());
      sh_transform_type source2SH(source_volumes.size(), Math::SH::NforL(lmax));
      sh_transform_type source2target(source_volumes.size(), target_volumes.size());
      data_vector_type predicted_data(target_volumes.size());

      if (pe_config.rows() == 2) {

        // Generate the direction set for the source data
        for (size_t source_index = 0; source_index != source_volumes.size(); ++source_index)
          Math::Sphere::cartesian2spherical(grad_in.block<1,3>(source_volumes[source_index], 0), source_dirset.row(source_index));
        // Generate the transformation from the source data to spherical harmonics
        // TODO For now, using the maximal spherical harmonic degree enabled by the source data
        // TODO For now, weighting all samples equally
        source2SH = Math::pinv(Math::SH::init_transform(source_dirset, lmax));
        // Compose transformation from source data to target data
        source2target = SH2target * source2SH;

        // Now we are ready to loop over the image
        Image<float> jacobian(jacobian_images[pe_index]);
        for (auto l = Loop(jacobian)(jacobian, dwi_in, dwi_out); l; ++l) {
          // How much weight are we attributing to the empirical data?
          // (if 1.0, we don't need to bother generating predictions)
          const default_type empirical_weight = std::max(1.0, default_type(jacobian.value()));
          if (empirical_weight == 1.0) {
            for (const auto volume : target_volumes) {
              dwi_in.index(3) = dwi_out.index(3) = volume;
              dwi_out.value() = dwi_in.value();
            }
          } else {
            // Grab the input data for generating the predictions
            for (size_t source_index = 0; source_index != source_volumes.size(); ++source_index) {
              dwi_in.index(3) = source_volumes[source_index];
              source_data[source_index] = dwi_in.value();
            }
            // Generate the predictions
            predicted_data = source2target * source_data;
            // Write these to the output image
            for (size_t target_index = 0; target_index != target_volumes.size(); ++target_index) {
              dwi_in.index(3) = dwi_out.index(3) = target_volumes[target_index];
              dwi_out.value() = (empirical_weight * dwi_in.value()) + ((1.0 - empirical_weight) * predicted_data[target_index]);
            }
          }
        }

      } else { // More than two phase encoding groups; therefore multiple phase encoding groups contributing to predictions

        data_vector_type source_weights(source_volumes.size());
        data_vector_type jacobians(pe_config.rows());

        // Build part of the requisite data for the A2SH transform in this voxel
        //   (the directions are the same for every voxel;
        //   the weights, which are influenced by the Jacobians, are not)
        for (size_t source_index = 0; source_index != source_volumes.size(); ++source_index)
          Math::Sphere::cartesian2spherical(grad_in.block<1,3>(source_volumes[source_index], 0), source_dirset.row(source_index));

        for (auto l = Loop(dwi_in, 0, 3)(dwi_in, dwi_out); l; ++l) {

          // We may need access to Jacobians for all phase encoding groups
          for (size_t jacobian_index = 0; jacobian_index != pe_config.rows(); ++jacobian_index) {
            assign_pos_of(dwi_in, 0, 3).to(jacobian_images[jacobian_index]);
            jacobians[jacobian_index] = jacobian_images[jacobian_index].value();
          }

          // If using exclusively empirical data,
          //   make that determination as soon as posibble to avoid unnecessary computation
          const default_type empirical_weight = std::max(1.0, jacobians[pe_index]);
          if (empirical_weight == 1.0) {
            for (const auto volume : target_volumes) {
              dwi_in.index(3) = dwi_out.index(3) = volume;
              dwi_out.value() = dwi_in.value();
            }
          } else {
            // Build the rest of the requisite data for the A2SH transform in this voxel
            // Also grab the input data while we're looping
            for (size_t source_index = 0; source_index != source_volumes.size(); ++source_index) {
              source_weights[source_index] = jacobians[pe_indices[source_volumes[source_index]]];
              dwi_in.index(3) = source_volumes[source_index];
              source_data[source_index] = dwi_in.value();
            }
            // Build the transformation from data in all other phase encoding groups to SH
            source2SH = Math::wls(Math::SH::init_transform(source_dirset, lmax), source_weights);
            // Compose transformation from source data to target data
            source2target = SH2target * source2SH;
            // Generate the predictions
            predicted_data = source2target * source_data;
            // Write these to the output image
            for (size_t target_index = 0; target_index != target_volumes.size(); ++target_index) {
              dwi_in.index(3) = dwi_out.index(3) = target_volumes[target_index];
              dwi_out.value() = (empirical_weight * dwi_in.value()) + ((1.0 - empirical_weight) * predicted_data[target_index]);
            }
          }
        }

      } // End branching on number of phase encoding groups being 2 or more

      ++progress;
    } // End looping over shells

  } // End looping over phase encoding groups

}



void run() {

  auto dwi_in = Header::open(argument[0]).get_image<float>();
  auto grad_in = DWI::get_DW_scheme(dwi_in);
  auto pe_in = PhaseEncoding::get_scheme(dwi_in);

  Header header_out(dwi_in);
  header_out.datatype() = DataType::Float32;
  header_out.name() = std::string(argument[2]);

  switch(int(argument[1])) {

   case 0: // combine_pairs
    run_combine_pairs(dwi_in, grad_in, pe_in, header_out);
    PhaseEncoding::clear_scheme(header_out);
    break;

   default: // no others yet implemented
    assert (false);

  }

  // Only do this for some operations
  //PhaseEncoding::clear_scheme(header);
  DWI::export_grad_commandline(header_out);



}
