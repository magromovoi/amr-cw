graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "team"
  ]
  node [
    id 1
    label "name"
  ]
  node [
    id 2
    label "city"
  ]
  node [
    id 3
    label "person"
  ]
  node [
    id 4
    label "arsenal"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 1
    target 4
    key 0
    label ":op1"
  ]
  edge [
    source 2
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 3
    target 1
    key 0
    label ":name"
  ]
]
