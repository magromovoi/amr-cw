graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "event"
  ]
  node [
    id 1
    label "name"
  ]
  node [
    id 2
    label "games"
  ]
  node [
    id 3
    label "city"
  ]
  node [
    id 4
    label "country"
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
