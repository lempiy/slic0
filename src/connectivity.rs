use std::collections::{HashMap, BTreeSet, HashSet};

pub fn create_new_state(state_length: usize) -> ConnectedComponentState {
  ConnectedComponentState{
    order: 0,
    relations: HashMap::new(),
    equality_relations: HashMap::new(),
    neighbours: HashMap::new(),
    state: vec![(0,0); state_length],
  }
}

pub struct ConnectedComponentState {
  pub order: u32,
  equality_relations: HashMap<u32,u32>,
  relations: HashMap<u32,ConnectedComponentRelation>,
  pub neighbours:  HashMap<u32, HashSet<u32>>,
  pub state: Vec<(usize,u32)>,
}

#[derive(PartialEq, Eq, Hash, Debug)]
struct ConnectedComponentRelation {
  equality: u32,
  centroid: Option<(u32,u32)>,
}

macro_rules! get_or_create_relation {
    ($relations:expr, $order: expr) => {
      if let Some(relation) = $relations.get_mut(&$order) {
        relation
      } else {
        $relations.insert($order, ConnectedComponentRelation{
          equality: $order,
          centroid: None,
        });
        $relations.get_mut(&$order).expect("new value not found")
      }
    };
}

impl ConnectedComponentState {
  pub fn set_equality(&mut self, order: u32, equals: u32) {
    let actual = self.resolve_equality(equals);
    let relation = get_or_create_relation!(self.relations, order);
    relation.equality = actual;
    self.equality_relations.insert(order, actual);
  }
  pub fn new_relation(&mut self) {
    self.order += 1;
    let relation = get_or_create_relation!(self.relations, self.order);
    self.equality_relations.insert(self.order, self.order);
  }
  pub fn set_pixel_state(&mut self, label_index: usize, state: (usize, u32)) {
    self.state[label_index] = state;
  }

  pub fn get_pixel_state(&self, label_index: usize) -> (usize, u32) {
    self.state[label_index]
  }

  pub fn resolve_equality(&self, order: u32) -> (u32) {
    let mut key = order;
    let mut result = *self.equality_relations.get(&key).expect("cannot find order in table");
    loop {
      key = result;
      result = *self.equality_relations.get(&result).expect("cannot find order in table");
      if result == key {
        break;
      }
    };
    result
  }
  pub fn mark_neighbours(&mut self, a: u32, b: u32) {
    self.mark_neighbour(a, b);
    self.mark_neighbour(b, a);
  }
  fn mark_neighbour(&mut self, key: u32, value: u32) {
    if let Some(set) = self.neighbours.get_mut(&key) {
      set.insert(value);
    } else {
      let mut set = HashSet::new();
      set.insert(value);
      self.neighbours.insert(key, set);
    }
  }
  pub fn debug(&self) {
    println!("{:?}", self.relations);
  }
}
