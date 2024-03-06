# Copyright (c) 2008-2023 the MRtrix3 contributors.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Covered Software is provided under this License on an "as is"
# basis, without warranty of any kind, either expressed, implied, or
# statutory, including, without limitation, warranties that the
# Covered Software is free of defects, merchantable, fit for a
# particular purpose or non-infringing.
# See the Mozilla Public License v. 2.0 for more details.
#
# For more details, see http://www.mrtrix.org/.

from mrtrix3 import algorithm, app, _version #pylint: disable=no-name-in-module

def usage(cmdline): #pylint: disable=unused-variable
  cmdline.set_author('Robert E. Smith (robert.smith@florey.edu.au)')
  cmdline.set_synopsis('Perform B1 field inhomogeneity correction for a DWI volume series')
  cmdline.add_description('Note that if the -mask command-line option is not specified, the MRtrix3 command dwi2mask will automatically be called to '
                          'derive a mask that will be passed to the relevant bias field estimation command. '
                          'More information on mask derivation from DWI data can be found at the following link: \n'
                          'https://mrtrix.readthedocs.io/en/' + _version.__tag__ + '/dwi_preprocessing/masking.html')
  common_options = cmdline.add_argument_group('Options common to all dwibiascorrect algorithms')
  common_options.add_argument('-mask', metavar='image', help='Manually provide a mask image for bias field estimation')
  common_options.add_argument('-bias', metavar='image', help='Output the estimated bias field')
  app.add_dwgrad_import_options(cmdline)

  # Import the command-line settings for all algorithms found in the relevant directory
  algorithm.usage(cmdline)
