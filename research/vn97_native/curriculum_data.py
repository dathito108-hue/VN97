"""Original generated tasks; split specification fixed before this run's scores."""
import hashlib
import json
import random

TASKS = ('copy', 'sum', 'deduplicate', 'classify')
TEMPLATES = {
 'train': {
  'copy': ['Trả lại mã {v}.', 'Sao chép số {v}.', 'Mã {v}; chỉ ghi lại mã.', 'Ghi nguyên mã: {v}.'],
  'sum': ['Cộng {a} và {b}.', 'Tính {a} + {b}.', 'Tổng hai số {a}, {b} là gì?', 'Cho kết quả cộng: {a}+{b}.'],
  'deduplicate': ['Bỏ trùng: {v}.', 'Giữ mỗi mục một lần: {v}.', 'Lọc mục lặp trong {v}.', 'Danh sách {v}; bỏ phần tử trùng.'],
  'classify': ['Loại yêu cầu: {v}.', 'Chọn nhãn cho yêu cầu {v}.', 'Nhãn tác vụ của {v} là gì?', 'Phân nhóm yêu cầu {v}.']},
 'validation': {
  'copy':['Viết lại đúng số mã {v}.'], 'sum':['Kết quả {a} cộng {b}?'],
  'deduplicate':['Danh sách không lặp của {v}?'], 'classify':['Gán nhãn yêu cầu: {v}.']},
 'test': {
  'copy':['Chỉ xuất mã đang có: {v}.'], 'sum':['Hãy cộng hai giá trị: {a} và {b}.'],
  'deduplicate':['Xuất các mục duy nhất theo thứ tự: {v}.'], 'classify':['Yêu cầu {v} thuộc nhãn nào?']}}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_splits():
    rng = random.Random(1097)
    used = {task:set() for task in TASKS}
    # Exclude earlier pilot cases from the new numeric pools as well.
    used['copy'].update(str(x) for a,b in ((100,228),(500,516),(900,916)) for x in range(a,b))
    used['sum'].update(str(sorted((i%40+1,i%11+1))) for i in range(128))
    used['sum'].update(str(sorted((41+i,12+i))) for i in range(16))
    result = {}
    for split, count in (('train',512), ('validation',32), ('test',32)):
        rows=[]
        for task in TASKS:
            for i in range(count):
                while True:
                    if task == 'copy':
                        value = str(rng.randrange(10_000))
                        fields, answer, case = {'v':value}, value, value
                    elif task == 'sum':
                        a,b = rng.randrange(100),rng.randrange(100)
                        fields, answer, case = {'a':a,'b':b}, str(a+b), str(sorted((a,b)))
                    elif task == 'deduplicate':
                        first,second = rng.sample(['a','b','c','d','e','f','g','h'],2)
                        number=str(rng.randrange(1000))
                        fields={'v':f'{first},{number},{first},{second},{number}'}
                        answer=f'{first},{number},{second}'
                        case=fields['v']
                    else:
                        index=i%4
                        # Category vocabulary necessarily repeats; template differs.
                        keyword=['csv','tổng','mã','danh mục'][index]
                        fields={'v':keyword}
                        answer=['clean_csv','sum','copy_id','catalog'][index]
                        case=f'{split}:{i}'
                    if case not in used[task]:
                        used[task].add(case)
                        break
                template_index = i // 4 if task == 'classify' else i
                template=TEMPLATES[split][task][template_index%len(TEMPLATES[split][task])]
                rows.append({'task':task,'case':case,'prompt':template.format(**fields)+'\nĐáp: ', 'answer':answer+'\n'})
        result[split]=rows
    # Exact prompt separation; task rules and output alphabet intentionally shared.
    seen=set()
    for rows in result.values():
        prompts={row['prompt'] for row in rows}
        if seen & prompts:
            raise ValueError('cross-split prompt leakage')
        seen |= prompts
    return result
