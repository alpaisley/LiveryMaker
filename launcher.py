"""Entry point for the packaged app and for running from source.
   Livery Maker                      -> opens the app
   Livery Maker --build file.lmproj  -> builds every livery in the project without the window"""
import os
import sys

# A windowed Windows build has no console: give print() somewhere harmless to write.
for _name in ('stdout', 'stderr'):
    if getattr(sys, _name) is None:
        setattr(sys, _name, open(os.devnull, 'w'))

if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--build':
        from livery_maker.cli import main as cli
        sys.exit(cli(sys.argv[2:]))
    from livery_maker.app import main
    main()
