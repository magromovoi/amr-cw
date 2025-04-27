graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "lead-02"
  ]
  node [
    id 1
    label "person"
  ]
  node [
    id 2
    label "political-party"
  ]
  node [
    id 3
    label "name"
  ]
  node [
    id 4
    label "mr"
  ]
  node [
    id 5
    label "government-organization"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":ARG0"
  ]
  edge [
    source 0
    target 2
    key 0
    label ":ARG1"
  ]
  edge [
    source 2
    target 3
    key 0
    label ":name"
  ]
  edge [
    source 3
    target 4
    key 0
    label ":op1"
  ]
  edge [
    source 5
    target 3
    key 0
    label ":name"
  ]
]
