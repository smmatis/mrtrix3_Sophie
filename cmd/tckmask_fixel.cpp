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

#include "algo/loop.h"
#include "command.h"
#include "image.h"
#include "progressbar.h"

#include "fixel/fixel.h"
#include "fixel/helpers.h"
#include "fixel/loop.h"

#include "dwi/tractography/mapping/loader.h"
#include "dwi/tractography/mapping/mapper_tckmaskfixel.h"
#include "dwi/tractography/mapping/writer.h"

#include "dwi/tractography/editing/editing.h"
#include "dwi/tractography/editing/loader.h"
#include "dwi/tractography/editing/receiver.h"
#include "dwi/tractography/editing/worker.h"

#include "ordered_thread_queue.h"

using namespace MR;
using namespace App;
using namespace MR::DWI;
using namespace MR::DWI::Tractography;
using namespace MR::DWI::Tractography::Editing;

using Fixel::index_type;

#define DEFAULT_ANGLE_THRESHOLD 10.0

// clang-format off
void usage() {

  AUTHOR = "Sophie Matis (sophie.matis@sydney.edu.au) adapted from David Raffelt (david.raffelt@florey.edu.au)";

  SYNOPSIS = "Mask a tractogram using a fixel folder.";

  EXAMPLES
  + Example ("Mask a track file to retain tracks with segments <5 degrees from any fixel",
             "tckmask_fixel -angle 5 original_track_file.tck fixel_folder track_file_out.tck",
             "This command uses the index and directions files from the fixel folder, " 
             "if you have a fixel mask to apply, first use the fixelcrop command to create a cropped fixel folder from that mask.");

  ARGUMENTS
  + Argument ("tracks",  "the input track file.").type_tracks_in()
  + Argument ("fixel_folder_in", "the input fixel folder;"
                                 "used to mask the tractogram").type_directory_in()
  + Argument ("tracks_out", "the output masked track file;"
                                  " this should be different to the input track file").type_tracks_out();

  OPTIONS
  + Option ("angle", "the max angle threshold for assigning streamline tangents to fixels"
                     " (default: " + str(DEFAULT_ANGLE_THRESHOLD, 2) + " degrees)")
    + Argument ("value").type_float(0.0, 90.0);
}
// clang-format on

void erase_if_present(Tractography::Properties &p, const std::string s) {
  auto i = p.find(s);
  if (i != p.end())
    p.erase(i);
}

void run() {
  const std::string input_fixel_folder = argument[1];
  Header index_header = Fixel::find_index_header(input_fixel_folder);
  auto index_image = index_header.get_image<index_type>();

  const index_type num_fixels = Fixel::get_number_of_fixels(index_header);

  const float angular_threshold = get_option_value("angle", DEFAULT_ANGLE_THRESHOLD);

  std::vector<Eigen::Vector3d> positions(num_fixels);
  std::vector<Eigen::Vector3d> directions(num_fixels);

  {
    auto directions_data = Fixel::find_directions_header(input_fixel_folder).get_image<default_type>().with_direct_io();
    // Load template fixel directions
    Transform image_transform(index_image);
    for (auto i = Loop("loading template fixel directions and positions", index_image, 0, 3)(index_image); i; ++i) {
      const Eigen::Vector3d vox(
          (default_type)index_image.index(0), (default_type)index_image.index(1), (default_type)index_image.index(2));
      index_image.index(3) = 1;
      index_type offset = index_image.value();
      index_type fixel_index = 0;
      for (auto f = Fixel::Loop(index_image)(directions_data); f; ++f, ++fixel_index) {
        directions[offset + fixel_index] = directions_data.row(1);
        positions[offset + fixel_index] = image_transform.voxel2scanner * vox;
      }
    }
  }

  std::vector<uint16_t> fixel_TDI(num_fixels, 0.0);
  const std::string track_filename = argument[0];
  const std::string output_track_filename = argument[2];
  DWI::Tractography::Properties properties;
  DWI::Tractography::Reader<float> track_file(track_filename, properties);
  // Read in tracts, and compute whole-brain fixel-fixel connectivity
  const size_t num_tracks = properties["count"].empty() ? 0 : to<int>(properties["count"]);
  if (!num_tracks)
    throw Exception("no tracks found in input file");

  {
    using SetVoxelDir = DWI::Tractography::Mapping::SetVoxelDir;
    //tckedit uses different loader (MR::DWI::Tractography::Editing::Loader)
    DWI::Tractography::Mapping::TrackLoader loader(track_file, num_tracks, "mapping tracks to fixels");
    DWI::Tractography::Mapping::TrackMapperFixels mapper(index_image, directions, angular_threshold);
    mapper.set_upsample_ratio(DWI::Tractography::Mapping::determine_upsample_ratio(index_header, properties, 0.333f));
    mapper.set_use_precise_mapping(true);
    Tractography::Properties properties;
    Editing::load_properties(properties);
    const size_t number = get_option_value("number", size_t(0));
    const size_t skip = get_option_value("skip", size_t(0));
    Receiver receiver(output_track_filename, properties, number, skip);
    Thread::run_ordered_queue(
      loader,
      Thread::batch(Streamline<float>()),
      mapper,
      Thread::batch(Streamline<float>()),
      receiver
    );
  }
  track_file.close();
}

