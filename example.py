from groupings import Cohort
from glob import glob
import time
import sys
import tools
import utils
import anytree
#from pathos.pools import ProcessPool
from concurrent.futures import ProcessPoolExecutor as ProcessPool
from concurrent.futures import ThreadPoolExecutor as ThreadPool

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



def apply_tools(dataset, tools):
    def _tool_fn(tools, path):
        def wrapper(mod):
            apply_tool(mod, tools, path)
            return mod
        return wrapper
    it = dataset.iter_modalities()
    fn = _tool_fn(tools, dataset.writepath)
    with ThreadPool() as P:
        modified = list(P.map(fn, it))
    ThreadPool().shutdown()


def apply_tool(dataset, tools, path=None):
    utils.colorwarn(f'Applied tools will write over reconstructed volume files')
    it = dataset.iter_volumes()
    if not path:
        path = dataset.writepath
    for volume in it:
        for name, reconfiles in volume.items():
            newfiles = []
            for f in reconfiles:
                for tool in tools:
                    f = tool(f, path)
                newfiles.append(f)
            volume[name] = newfiles


toolset = [tools.Interpolate()]
print('applying tools')
apply_tools(cohort, toolset)
print(cohort)


print('elapsed:', time.time() - start)
print(len(cohort))
sys.stdout.flush()