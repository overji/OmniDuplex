"""One public entrypoint for generation, rendering and merging."""
import importlib
import sys


COMMANDS = {
    'pipeline': 'pipeline', 'sample': 'generate', 'batch': 'batch',
    'render-prompts': 'video_caption', 'merge': 'merge_qa',
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv == ['--help'] or argv == ['-h']:
        print('Usage: generate.py [pipeline] --videos-dir DIR --output-dir DIR --num-attempts N [API options]')
        print('Commands: ' + ', '.join(COMMANDS))
        print('Use generate.py COMMAND --help for command-specific options.')
        return 0
    command = argv.pop(0) if argv[0] in COMMANDS else 'pipeline'
    if argv and not argv[0].startswith('-') and command == 'pipeline':
        print(f'error: unknown command {argv[0]!r}', file=sys.stderr)
        return 2
    return importlib.import_module('.' + COMMANDS[command], __package__).main(argv) or 0
