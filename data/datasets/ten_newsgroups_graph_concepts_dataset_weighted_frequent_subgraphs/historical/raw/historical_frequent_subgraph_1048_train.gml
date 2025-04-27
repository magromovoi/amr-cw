graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "military"
  ]
  node [
    id 1
    label "name"
  ]
  node [
    id 2
    label "person"
  ]
  node [
    id 3
    label "world-region"
  ]
  node [
    id 4
    label "country"
  ]
  node [
    id 5
    label "and"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 2
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 2
    target 4
    key 0
    label ":mod"
  ]
  edge [
    source 3
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 4
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 5
    target 2
    key 0
    label ":op1"
  ]
  edge [
    source 5
    target 4
    key 0
    label ":op1"
  ]
]
