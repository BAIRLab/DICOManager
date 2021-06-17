from groupings import Cohort
from glob import glob
import time
import sys
import tools
import utils
import anytree

'''
# in the format of:
filter_list = {'PatientID': [...],
               'StudyDate': [...],
               'SeriesDate': [...],
               'StructName': [...]}
'''
filter_list = {'StructName': {'hippocampus': ['hippocampus'],
                              'hippo_avoid': ['hippoavoid', 'hippo_avoid']},
               'Modality': ['CT', 'RTSTRUCT']}

start = time.time()
files = glob('/home/eporter/eporter_data/rtog_project/MIMExport/**/*.dcm', recursive=True)
cohort = Cohort(name='RTOG_Hippocampus', files=files, include_series=False, filter_by=filter_list)
print(cohort)
#cohort.save_tree(path='/home/eporter/eporter_data/rtog_project/dicoms/')
print('tree saved, now reconstructing')
cohort.recon(parallelize=True, in_memory=False, path='/home/eporter/eporter_data/rtog_project/built/')

"""
def _tool_fn(tools):
    def wrapper(mod):
        print('starting')
        for volname, vol in mod.volumes_data.items():
            print(volname)
            for name, filelist in vol.items():
                print(name)
                newfiles = []
                while len(filelist) > 0:
                    f = filelist.pop()
                    for tool in tools:
                        f = tool(f)
                    newfiles.append(f)
                vol[name] = newfiles
        return mod
    return wrapper


def replace_nodes(tree, node):
    print('replacing')
    node.name += 'UPDATED'
    found = anytree.search.find(tree, filter_=lambda x: x.name == node.name)
    found.volumes = node.volumes


def apply_tools(dataset, tools):
    it = dataset.iter_modalities()
    fn = _tool_fn(tools)
    ProcessPool().clear()
    with ProcessPool() as P:
        print('test')
        modified = list(P.map(fn, it))
    ProcessPool().clear()
    for node in modified:
        replace_nodes(dataset, node)
"""

def apply_tools(dataset, tools):
    utils.colorwarn(f'Applied tools will write over reconstructed volume files')
    it = dataset.iter_volumes()
    for volume in it:
        for name, reconfiles in volume.items():
            newfiles = []
            for f in reconfiles:
                for tool in tools:
                    f = tool(f)
                newfiles.append(f)
            volume[name] = newfiles


#toolset = [tools.interpolate]
toolset = [tools.Interpolate()]
print('applying tools')
apply_tools(cohort, toolset)
print(cohort)

print('elapsed:', time.time() - start)
print(len(cohort))
sys.stdout.flush()