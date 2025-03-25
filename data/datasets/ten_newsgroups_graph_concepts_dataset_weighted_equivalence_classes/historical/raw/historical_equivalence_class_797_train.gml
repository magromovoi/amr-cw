graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "person"
  ]
  node [
    id 1
    label "country"
  ]
  node [
    id 2
    label "troop"
  ]
  node [
    id 3
    label "and"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":mod"
  ]
  edge [
    source 2
    target 1
    key 0
    label ":mod"
  ]
  edge [
    source 3
    target 0
    key 0
    label ":op2"
  ]
]
