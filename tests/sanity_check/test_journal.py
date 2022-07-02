#!/usr/bin/env python

def test1():
    from mcni import journal
    logger = journal.logger(
        'info', 'test', header='', footer='', format=' | {}')
    import journal
    info = journal.info("test")
    info.activate()
    logger('hello')
    return

def test2():
    import journal
    info = journal.info("test")
    info.activate()
    info.log('hello')
    return

def main():
    test1()
    test2()

if __name__ == '__main__': main()
