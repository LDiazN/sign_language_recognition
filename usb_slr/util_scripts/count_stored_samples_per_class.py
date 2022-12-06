# Use this script to compute how many instances per class you have

from slr.dataset_manager.dataset_managers import MicrosoftDatasetManager

dm = MicrosoftDatasetManager()
data = dm.train_numeric_dataset_client.retrieve_data()

label_count = {}
for (_, desc) in data:

    if desc.label not in label_count:
        label_count[desc.label] = 0

    label_count[desc.label] += 1

# convert label count to list and sort by count

label_count = list(label_count.items())
label_count.sort(key=lambda x: x[1], reverse=True)

for (l, c) in label_count[:100]:
    print(f"{l} : {c}")

3 : 35
19 : 32
15 : 30
8 : 29
33 : 29
11 : 29
1 : 28
14 : 28
12 : 28
51 : 27
50 : 27
2 : 27
29 : 27
79 : 27
23 : 26
78 : 26
9 : 26
31 : 26
61 : 25
25 : 25
10 : 25
7 : 25
44 : 24
17 : 24
75 : 24
39 : 24
48 : 24
59 : 24
21 : 23
45 : 23
65 : 23
95 : 23
76 : 23
34 : 23
64 : 23
32 : 23
13 : 23
83 : 23
66 : 23
92 : 23
47 : 22
43 : 22
52 : 22
72 : 22
24 : 22
28 : 22
6 : 22
71 : 22
26 : 22
99 : 22
16 : 22
41 : 21
20 : 21
37 : 21
110 : 21
22 : 21
38 : 21
171 : 21
219 : 20
121 : 20
30 : 20
118 : 20
42 : 20
74 : 20
63 : 20
123 : 19
27 : 19
129 : 19
55 : 19
58 : 19
117 : 19
67 : 19
77 : 19
109 : 19
96 : 18
342 : 18
35 : 18
100 : 18
54 : 18
53 : 18
105 : 18
91 : 18
245 : 18
97 : 18
212 : 18
62 : 18
154 : 18
36 : 18
265 : 18
103 : 18
230 : 18
163 : 17
107 : 17
255 : 17
82 : 17
239 : 17
276 : 17
132 : 17
127 : 17
122 : 17