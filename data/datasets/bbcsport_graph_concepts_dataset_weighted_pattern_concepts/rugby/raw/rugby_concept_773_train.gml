graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "person"
  ]
  node [
    id 1
    label "name"
  ]
  node [
    id 2
    label "coach-01"
  ]
  node [
    id 3
    label "team"
  ]
  node [
    id 4
    label "and"
  ]
  node [
    id 5
    label "country"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 2
    target 0
    key 0
    label ":ARG0"
  ]
  edge [
    source 2
    target 3
    key 0
    label ":ARG1"
  ]
  edge [
    source 3
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 4
    target 0
    key 0
    label ":op2"
  ]
  edge [
    source 4
    target 0
    key 1
    label ":op1"
  ]
  edge [
    source 4
    target 0
    key 2
    label ":op3"
  ]
  edge [
    source 5
    target 1
    key 0
    label ":name"
  ]
]
