graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "league"
  ]
  node [
    id 1
    label "name"
  ]
  node [
    id 2
    label "chelsea"
  ]
  node [
    id 3
    label "team"
  ]
  node [
    id 4
    label "person"
  ]
  node [
    id 5
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
    target 2
    key 0
    label ":op1"
  ]
  edge [
    source 1
    target 5
    key 0
    label ":op1"
  ]
  edge [
    source 1
    target 0
    key 0
    label ":op2"
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
]
