import datetime
import hashlib
import json
import pathlib
import re
import subprocess


base = pathlib.Path('/home/syk/worktrees/sgl-kernel-xpu-up')
artifacts = base / 'muse-upstream-artifacts-20261017'
modules = {'bf16-gemm': 'muse-bf16-gemm-xe35', 'moe-permutation': 'muse-moe-permutation-xe35'}
data = {}
for module, tree_name in modules.items():
    tree = base / 'pr' / tree_name
    build = artifacts / module / 'full-default-build'
    container_tree = str(tree).replace('/home/syk/', '/root/')
    container_build = str(build).replace('/home/syk/', '/root/')

    def normalize(text):
        return text.replace(container_tree, '<SOURCE>').replace(container_build, '<BUILD>')

    files = subprocess.check_output(['git', 'ls-files', '-c', '-o', '--exclude-standard', '-z'], cwd=tree).decode().split('\0')
    hashes = {name: hashlib.sha256((tree / name).read_bytes()).hexdigest() for name in files if name and (tree / name).is_file()}
    ninja = (build / 'build.ninja').read_text()
    edges = {}
    for block in re.split(r'\n(?=build )', ninja):
        if not block.startswith('build '):
            continue
        lines = block.splitlines()
        block = '\n'.join(line for line in lines if line.startswith('build ') or line.startswith('  '))
        output = lines[0].split(': ', 1)[0][6:]
        rule = lines[0].split(': ', 1)[1].split()[0]
        if rule != 'phony' and (output.endswith('.o') or output.endswith('.so')):
            edges[output] = normalize(block)
    scripts = {}
    dependencies = {}
    for match in re.finditer(r'-P ([^\s]+\.Release\.cmake)', ninja):
        script = pathlib.Path(match.group(1).replace('/root/', '/home/syk/', 1))
        if script.is_file():
            relative = str(script.relative_to(build))
            scripts[relative] = normalize(script.read_text())
            depfile = pathlib.Path(str(script).replace('.Release.cmake', '.depend'))
            if depfile.exists():
                content = depfile.read_text()
                paths = re.findall(r'"([^"\n]+)"', content)
                dependencies[relative] = [normalize(path) for path in paths]
    generated = {str(path.relative_to(build)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (build / 'src' / 'generated').rglob('*.cpp')}
    data[module] = {'source_hashes': hashes, 'edges': edges, 'scripts': scripts, 'dependencies': dependencies, 'generated_hashes': generated}

first, second = data.values()
common_files = set(first['source_hashes']) & set(second['source_hashes'])
changed = sorted(name for name in common_files if first['source_hashes'][name] != second['source_hashes'][name])
summary = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'no_build_or_gpu_commands': True,
           'source_common_count': len(common_files), 'source_identical_count': len(common_files) - len(changed), 'source_changed': changed,
           'source_only': {module: sorted(set(item['source_hashes']) - common_files) for module, item in data.items()}}
for category in ['edges', 'scripts', 'generated_hashes']:
    common = set(first[category]) & set(second[category])
    different = sorted(name for name in common if first[category][name] != second[category][name])
    summary[category] = {'counts': {module: len(item[category]) for module, item in data.items()}, 'common': len(common), 'equal': len(common) - len(different), 'different': different,
                         'only': {module: sorted(set(item[category]) - common) for module, item in data.items()}}
summary['dependency_samples'] = {}
for module, item in data.items():
    populated = {name: paths for name, paths in item['dependencies'].items() if paths}
    summary['dependency_samples'][module] = {'populated': len(populated), 'unpopulated': len(item['dependencies']) - len(populated),
                                           'generic_ops_header_consumers': sorted(name for name, paths in populated.items() if '<SOURCE>/include/sgl_kernel_ops.h' in paths),
                                           'changed_common_header_consumers': sorted(name for name, paths in populated.items() if any(path == '<SOURCE>/' + name_changed for path in paths for name_changed in changed if name_changed.endswith(('.h', '.hpp'))))}
summary['command_comparison'] = {}
common_edges = set(first['edges']) & set(second['edges'])
changed_commands = []
dependency_only_changes = []
for name in sorted(common_edges):
    first_values = [line for line in first['edges'][name].splitlines() if line.startswith('  ')]
    second_values = [line for line in second['edges'][name].splitlines() if line.startswith('  ')]
    if first_values != second_values:
        changed_commands.append(name)
    elif first['edges'][name] != second['edges'][name]:
        first_paths = set(first['edges'][name].splitlines()[0].split())
        second_paths = set(second['edges'][name].splitlines()[0].split())
        dependency_only_changes.append({'edge': name, 'first_only_count': len(first_paths - second_paths), 'second_only_count': len(second_paths - first_paths), 'changed_source_dependencies': sorted(path for path in first_paths ^ second_paths if path.startswith('<SOURCE>'))})
summary['command_comparison'] = {'common': len(common_edges), 'identical_command_variables': len(common_edges) - len(changed_commands), 'different_command_variables': changed_commands, 'dependency_only_changes': dependency_only_changes}
output = artifacts / 'bf16-gemm' / 'cache-reuse-readonly-evidence.json'
output.write_text(json.dumps({'summary': summary, 'detail': data}, indent=2) + '\n')
print(json.dumps(summary, indent=2))
