graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "multi-sentence"
  ]
  node [
    id 1
    label "and"
  ]
  node [
    id 2
    label "add-02"
  ]
  node [
    id 3
    label "you"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":snt1"
  ]
  edge [
    source 2
    target 1
    key 0
    label ":ARG1"
  ]
  edge [
    source 2
    target 3
    key 0
    label ":ARG0"
  ]
]
