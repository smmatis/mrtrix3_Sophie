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


import math, os, re, shutil
from mrtrix3 import CONFIG
from mrtrix3 import app

POSTFIXES = [ 'B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB' ]

def execute(): #pylint: disable=unused-variable

  file_regex = re.compile(r"^mrtrix-tmp-[a-zA-Z0-9]{6}\..*$")
  file_config_regex = re.compile(r"^" + CONFIG['TmpFilePrefix'] + r"[a-zA-Z0-9]{6}\..*$") \
                      if 'TmpFilePrefix' in CONFIG and CONFIG['TmpFilePrefix'] != 'mrtrix-tmp-' \
                      else None
  dir_regex = re.compile(r"^\w+-tmp-[a-zA-Z0-9]{6}$")
  dir_config_regex = re.compile(r"^" + CONFIG['ScriptScratchPrefix'] + r"[a-zA-Z0-9]{6}$") \
                      if 'ScriptScratchPrefix' in CONFIG \
                      else None

  files_to_delete = [ ]
  dirs_to_delete = [ ]
  root_dir = os.path.abspath(app.ARGS.path)
  print_search_dir = ('' if os.path.abspath(os.getcwd()) == root_dir else ' from ' + root_dir)
  def file_search(regex):
    files_to_delete.extend([ os.path.join(dirname, filename) for filename in filter(regex.search, filelist) ])
  def dir_search(regex):
    items = set(filter(regex.search, subdirlist))
    if items:
      dirs_to_delete.extend([os.path.join(dirname, subdirname) for subdirname in items])
      subdirlist[:] = list(set(subdirlist)-items)
  def print_msg():
    return 'Searching' + print_search_dir + ' (found ' + str(len(files_to_delete)) + ' files, ' + str(len(dirs_to_delete)) + ' directories)'
  progress = app.ProgressBar(print_msg)
  for dirname, subdirlist, filelist in os.walk(root_dir):
    file_search(file_regex)
    if file_config_regex:
      file_search(file_config_regex)
    dir_search(dir_regex)
    if dir_config_regex:
      dir_search(dir_config_regex)
    progress.increment()
  progress.done()

  if app.ARGS.test:
    if files_to_delete:
      app.console('Files identified (' + str(len(files_to_delete)) + '):')
      for filepath in files_to_delete:
        app.console('  ' + filepath)
    else:
      app.console('No files' + ('' if dirs_to_delete else ' or directories') + ' found')
    if dirs_to_delete:
      app.console('Directories identified (' + str(len(dirs_to_delete)) + '):')
      for dirpath in dirs_to_delete:
        app.console('  ' + dirpath)
    elif files_to_delete:
      app.console('No directories identified')
  elif files_to_delete or dirs_to_delete:
    progress = app.ProgressBar('Deleting temporaries (' + str(len(files_to_delete)) + ' files, ' + str(len(dirs_to_delete)) + ' directories)', len(files_to_delete) + len(dirs_to_delete))
    except_list = [ ]
    size_deleted = 0
    for filepath in files_to_delete:
      filesize = 0
      try:
        filesize = os.path.getsize(filepath)
        os.remove(filepath)
        size_deleted += filesize
      except OSError:
        except_list.append(filepath)
      progress.increment()
    for dirpath in dirs_to_delete:
      dirsize = 0
      try:
        for dirname, subdirlist, filelist in os.walk(dirpath):
          dirsize += sum(os.path.getsize(filename) for filename in filelist)
      except OSError:
        pass
      try:
        shutil.rmtree(dirpath)
        size_deleted += dirsize
      except OSError:
        except_list.append(dirpath)
      progress.increment()
    progress.done()
    postfix_index = int(math.floor(math.log(size_deleted, 1024))) if size_deleted else 0
    if postfix_index:
      size_deleted = round(size_deleted / math.pow(1024, postfix_index), 2)
    def print_freed():
      return ' (' + str(size_deleted) + POSTFIXES[postfix_index] + ' freed)' if size_deleted else ''
    if except_list:
      app.console(str(len(files_to_delete) + len(dirs_to_delete) - len(except_list)) + ' of ' + str(len(files_to_delete) + len(dirs_to_delete)) + ' items erased' + print_freed())
      if app.ARGS.failed:
        with open(app.ARGS.failed, 'w', encoding='utf-8') as outfile:
          for item in except_list:
            outfile.write(item + '\n')
        app.console('List of items script failed to erase written to file "' + app.ARGS.failed + '"')
      else:
        app.console('Items that could not be erased:')
        for item in except_list:
          app.console('  ' + item)
    else:
      app.console('All items deleted successfully' + print_freed())
  else:
    app.console('No files or directories found')
