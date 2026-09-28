import unittest
from curriculum_data import TASKS, build_splits, digest


class CurriculumTests(unittest.TestCase):
    def test_stable_balanced_disjoint_cases(self):
        splits=build_splits()
        self.assertEqual(digest(splits),digest(build_splits()))
        self.assertEqual([len(v) for v in splits.values()],[2048,128,128])
        for task in TASKS:
            seen=set()
            for split, rows in splits.items():
                selected=[r for r in rows if r['task']==task]
                self.assertEqual(len(selected),512 if split=='train' else 32)
                cases={r['case'] for r in selected}
                self.assertFalse(seen & cases)
                seen |= cases
                for r in selected:
                    self.assertLessEqual(len((r['prompt']+r['answer']).encode())-1,128)
                    self.assertTrue(r['answer'].endswith('\n'))
                    if task=='copy':
                        self.assertFalse(any(a<=int(r['case'])<b for a,b in ((100,228),(500,516),(900,916))))
                    if task=='deduplicate':
                        values=r['case'].split(',')
                        self.assertEqual(r['answer'].strip(),','.join(dict.fromkeys(values)))


if __name__=='__main__':
    unittest.main()
