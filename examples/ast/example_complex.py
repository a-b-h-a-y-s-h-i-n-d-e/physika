EXPECTED = {'functions': {},
 'classes': {},
 'program': [('decl', 'x', 'ℂ', ('add', ('num', 3), ('num', 1j)), 1),
             ('decl', 'y', 'ℂ', ('add', ('num', 5), ('num', 3j)), 2),
             ('expr', ('var', 'x'), 0),
             ('expr', ('var', 'y'), 0),
             ('expr', ('add', ('var', 'x'), ('var', 'y')), 0)]}
