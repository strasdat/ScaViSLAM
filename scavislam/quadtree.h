// This file is part of ScaViSLAM.
//
// Copyright 2011 Hauke Strasdat (Imperial College London)
//
// ScaViSLAM is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// any later version.
//
// ScaViSLAM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with ScaViSLAM.  If not, see <http://www.gnu.org/licenses/>.

#ifndef SCAVISLAM_QUADTREE_H
#define SCAVISLAM_QUADTREE_H

#include <queue>
#include <stack>
#include <tr1/array>

#include <opencv2/opencv.hpp>

#include <visiontools/sample.h>

#include "global.h"
#include "maths_utils.h"

namespace ScaViSLAM
{

using namespace std;
using namespace Eigen;

//TODO: clean, hide implementation from definition

template <typename T>
struct QuadTreeElement
{
  QuadTreeElement()
  {
  }

  QuadTreeElement(const Vector2d & pos,
                   const T & t): pos(pos), content(t)
  {
  }

  Vector2d pos;
  T content;
};

template <typename T>
class QuadTree;

template <typename T>
class QuadTreeNode
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW


  friend class QuadTree<T>;
  class EquiIter;
  friend class EquiIter;

  QuadTreeNode(const Rectangle & bbox,
               int unique_id,
               QuadTree<T> *  qtree)
    :
      children_(NULL),
      unique_id_(unique_id),
      bbox_(bbox),
      is_empty_(true),
      qtree_(qtree)
  {
  }

  ~QuadTreeNode()
  {
    delete children_;
  }

  static inline bool
  intersects(const Rectangle & A,
             const Rectangle & B);

  bool
  deleteAtPos(const Vector2d & pos);

  bool
  insert(const QuadTreeElement<T> & new_elemz);

  void
  query(const Rectangle & win,
        typename ALIGNED<QuadTreeElement<T> >::list * content_list) const;
  bool
  isWindowEmpty(const Rectangle & win) const;
  //Mainly for debugging:
  //Shows the whole tree -- all nodes (quads) and leafs (elems)
  void
  traverse(list<QuadTreeElement<T> > & elem_list,
           list<Rectangle> & quad_list) const;
  int unique_id() const
  {
    return unique_id_;
  }
  bool is_empty() const
  {
    return is_empty_;
  }

  QuadTreeElement<T> elem;

private:
  struct Children
  {
    Children(QuadTreeNode * xy,
             QuadTreeNode * xY,
             QuadTreeNode * Xy,
             QuadTreeNode * XY)
      : xy(xy), xY(xY), Xy(Xy), XY(XY)
    {
    }

    ~Children()
    {
      delete xy;
      delete xY;
      delete Xy;
      delete XY;
    }
    QuadTreeNode * xy;
    QuadTreeNode * xY;
    QuadTreeNode * Xy;
    QuadTreeNode * XY;

    inline bool insert(const QuadTreeElement<T> & elem,
                       const Rectangle & bbox);
  };

  Children * children_;
  int unique_id_;
  Rectangle bbox_;
  bool is_empty_;
  QuadTree<T> *  qtree_;
};


template <typename T> class QuadTree
{
  friend class QuadTreeNode<T>;

public:


  class EquiIter;
  friend class EquiIter;

  class  EquiIter : public iterator<forward_iterator_tag,
      QuadTreeElement<T> >
  {
    const QuadTreeElement<T> * ptr;
    map<int,vector<const QuadTreeNode<T>* >  > node_queue;
    bool _reached_end;
    tr1::unordered_set<int> visited;

    inline void add_permuted_children(
        const typename QuadTreeNode<T>::Children * ch,
        stack<const QuadTreeNode<T>* > & dfs_stack)
    {
      tr1::array<const QuadTreeNode<T> *,4> cur_it;
      cur_it[0] = ch->xy;
      cur_it[1] = ch->xY;
      cur_it[2] = ch->Xy;
      cur_it[3] = ch->XY;

      for (int i=0; i<4; ++i)
      {
        int k = Sample::uniform(0,3);
        const QuadTreeNode<T>* tmp = cur_it[k];
        cur_it[k] = cur_it[i];
        cur_it[i] = tmp;
      }

      dfs_stack.push((cur_it[0]));
      dfs_stack.push((cur_it[1]));
      dfs_stack.push((cur_it[2]));
      dfs_stack.push((cur_it[3]));
    }

  public:

    EquiIter(const EquiIter & bfs_it)
      : ptr(bfs_it.ptr),
        node_queue(bfs_it.node_queue),
        _reached_end(bfs_it._reached_end),
        visited(visited)
    {
    }

    EquiIter& operator= (EquiIter const& rhs)
    {
      ptr = rhs.ptr;
      node_queue = rhs.node_queue;
      _reached_end = rhs._reached_end;
      visited = rhs.visited;

    }

    EquiIter(const QuadTreeNode<T> & node)
      : ptr(0),
        _reached_end(false)

    {
      _reached_end = false;
      vector<const QuadTreeNode<T>* >  vec;
      vec.push_back(&node);

      node_queue.insert(make_pair(0,vec));
      operator++();
    }

    const QuadTreeElement<T>& operator*() const {
      return *ptr;
    }

    const QuadTreeElement<T> * operator->() const {
      return ptr;
    }

    bool operator==(const EquiIter& rhs)
    {
      return ptr==rhs.ptr;
    }

    bool operator!=(const EquiIter& rhs)
    {
      return ptr!=rhs.ptr;
    }

    bool reached_end()
    {
      return _reached_end;
    }

    EquiIter& operator++()
    {
      while (node_queue.size()>0)
      {
        typename map<int,vector<const QuadTreeNode<T>* >  >
            ::iterator level_queue_pair
            = node_queue.begin();

        vector<const QuadTreeNode<T>* >  & vec = level_queue_pair->second;
        int level = level_queue_pair->first;

        if (vec.size()==0)
        {
          node_queue.erase(level);
          continue;
        }

        int idx = Sample::uniform(0,vec.size()-1);

        const QuadTreeNode<T> * n = vec[idx];//.front();

        vec.erase(vec.begin()+idx);
        if (n->children_ == 0)
        {
          if (!n->is_empty() && visited.find(n->unique_id())==visited.end())
          {
            ptr = &(n->elem);
            return *this;
          }

        }
        else
        {
          //Do DFS to find 1 entry!
          stack<const QuadTreeNode<T>* > dfs_stack;
          const typename QuadTreeNode<T>::Children * initial_ch
              = n->children_;
          int cur_level= level+1;

          typename map<int,vector<const QuadTreeNode<T>* >  >
              ::iterator qu_it
              = node_queue.find(cur_level);
          if (qu_it==node_queue.end())
          {
            vector<const QuadTreeNode<T>* >  new_vec;
            qu_it = node_queue.insert(make_pair(cur_level,
                                                new_vec)).first;
          }
          qu_it->second.push_back(initial_ch->xy);
          qu_it->second.push_back(initial_ch->xY);
          qu_it->second.push_back(initial_ch->Xy);
          qu_it->second.push_back(initial_ch->XY);
          add_permuted_children(initial_ch, dfs_stack);
          while(dfs_stack.size()>0)
          {
            const QuadTreeNode<T> * n = dfs_stack.top();
            dfs_stack.pop();
            if(n->children_==0)
            {
              if(!n->is_empty() && visited.find(n->unique_id())==visited.end())
              {
                ptr = &(n->elem);
                visited.insert(n->unique_id());



                return *this;
              }
            }
            else
            {
              add_permuted_children(n->children_, dfs_stack);
            }
          }
        }
      }
      _reached_end = true;

      return *this;
    }

    EquiIter operator++(int)
    {
      EquiIter tmp(*this);
      operator++();
      return tmp;
    }

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  };

  class BfsIter;
  friend class BfsIter;

  class  BfsIter : public iterator<forward_iterator_tag,
      QuadTreeElement<T> >
  {
    const QuadTreeElement<T> * ptr;
    queue<const QuadTreeNode<T> *> node_queue;
    bool _reached_end;

  public:
    BfsIter(const BfsIter & bfs_it)
      : ptr(bfs_it.ptr),
        node_queue(bfs_it.node_queue),
        _reached_end(bfs_it._reached_end)
    {
    }

    BfsIter& operator= (BfsIter const& rhs)
    {
      ptr = rhs.ptr;
      node_queue = rhs.node_queue;
      _reached_end = rhs._reached_end;
    }

    BfsIter(const QuadTreeNode<T> & node)
      : ptr(0),
        _reached_end(false)

    {
      _reached_end = false;
      node_queue.push(&node);
      operator++();
    }

    const QuadTreeElement<T>& operator*() const {
      return *ptr;
    }

    const QuadTreeElement<T> * operator->() const {
      return ptr;
    }

    bool operator==(const BfsIter& rhs)
    {
      return ptr==rhs.ptr;
    }

    bool operator!=(const BfsIter& rhs)
    {
      return ptr!=rhs.ptr;
    }


    bool reached_end()
    {
      return _reached_end;
    }

    BfsIter& operator++()
    {
      while (node_queue.size()>0)
      {
        const QuadTreeNode<T> * n = node_queue.front();
        node_queue.pop();
        if (n->children_ == 0)
        {
          if (!n->is_empty() )
          {
            ptr = &(n->elem);
            return *this;
          }
        }
        else
        {
          node_queue.push(n->children_->xy);
          node_queue.push(n->children_->xY);
          node_queue.push(n->children_->Xy);
          node_queue.push(n->children_->XY);
        }
      }
      _reached_end = true;

      return *this;
    }

    BfsIter operator++(int)
    {
      BfsIter tmp(*this);
      operator++();
      return tmp;
    }
  };

public:
  QuadTree() :
    ids(-1),
    bbox(Rectangle(-1,-1,-1,-1)),

    root( bbox,get_new_id(),const_cast<QuadTree<T> *>(this)),
    delta(-1)
  {
  }

  QuadTree(const Rectangle & bbox,
           double delta)
    : ids(-1),
      bbox(bbox),
      root( bbox,get_new_id(),const_cast<QuadTree<T> *>(this)),
      delta(delta)
  {
  }

  EquiIter begin_equi() const
  {
    return EquiIter(root);
  }

  BfsIter begin_bfs() const
  {

    return BfsIter(root);
  }

  bool  insert(const Vector2d & pos,
               const T & t)
  {
    return root.insert(QuadTreeElement<T>(pos,t));
  }

  bool delete_at_pos( const Vector2d & p)
  {
    return root.deleteAtPos(p);
  }

  void query(const Rectangle & win,
             typename ALIGNED<QuadTreeElement<T> >::list * l) const
  {
    root.query(win,l);
  }

  bool isWindowEmpty(const Rectangle & win) const
  {
    return root.isWindowEmpty(win);
  }

  void traverse(list<QuadTreeElement<T> > & elem_list,
                list<Rectangle> & quad_list) const
  {
    return root.traverse(elem_list, quad_list);
  }

private:

  int get_new_id()
  {
    ++ids;
    return ids;
  }
  int ids;

  Rectangle bbox;
  QuadTreeNode<T>  root;
  double delta;

};

template<typename T>
bool QuadTreeNode<T>::Children::
insert(const QuadTreeElement<T> & elem,
       const Rectangle & bbox)
{
  double x_diff = bbox.width;
  double y_diff = bbox.height;
  double rel_x = 1-(bbox.x + bbox.width - elem.pos[0])/x_diff;
  double rel_y = 1-(bbox.y + bbox.height - elem.pos[1])/y_diff;

  assert(rel_x >= 0);
  assert(rel_y >= 0);
  assert(rel_x <= 1);
  assert(rel_y <= 1);

  if(rel_x < 0.5 && rel_y < 0.5)
  {
    return xy->insert(elem);
  }
  else if (rel_x >= 0.5 && rel_y < 0.5)
  {
    return Xy->insert(elem);
  }
  else if (rel_x < 0.5 && rel_y >= 0.5)
  {
    return xY->insert(elem);
  }
  else if (rel_x >= 0.5 && rel_y >= 0.5)
  {
    return XY->insert(elem);
  }

  assert(false);
  return false;
}

template<typename T>
bool QuadTreeNode<T>::
intersects(const Rectangle & A,
           const Rectangle & B)
{
  if(A.y+A.height <= B.y)
    return false;
  if(A.y    >= B.y+B.height)
    return false;
  if(A.x+A.width <= B.x)
    return false;
  if(A.x >= B.x+B.width)
    return false;
  return true;
}

template<typename T>
bool QuadTreeNode<T>::deleteAtPos(const Vector2d & pos)
{
  if (children_.size() == 0)
  {
    if(is_empty_)
    {
      return false;
    }
    else
    {
      if(norm(elem.pos-pos) < qtree_->delta)
      {
        is_empty_ = true;
        return true;
      }
      return false;
    }
  }
  else
  {
    assert(children_.size()==4);
    typename list<QuadTreeNode<T> >::iterator it =  children_.begin();
    if (it->bbox.contains(pos)){
      it->deleteAtPos(pos);
    }
    else
    {
      ++it;
      if (it->bbox.contains(pos)){
        it->deleteAtPos(pos);
      }
      else
      {
        ++it;
        if (it->bbox.contains(pos)){
          it->deleteAtPos(pos);
        }
        else
        {
          ++it;
          if (it->bbox.contains(pos)){
            it->deleteAtPos(pos);
          }
        }
      }
    }
    assert(false);
    return false;
  }
}

template<typename T>
bool QuadTreeNode<T>::insert(const QuadTreeElement<T> & new_elem)
{
  assert(bbox_.contains(cv::Point2d(new_elem.pos[0],new_elem.pos[1])));
  assert(isnan(new_elem.pos[0])==false );
  assert(isnan(new_elem.pos[1])==false );
  bool inserted ;
  if (children_ == 0)
  {
    if(is_empty_)
    {
      elem = new_elem;
      is_empty_ = false;
      inserted = true;
    }
    else
    {
      if ((elem.pos-new_elem.pos).norm()<qtree_->delta)
      {
        return false;
      }

      double x_diff = bbox_.width;
      double y_diff = bbox_.height;
      double x0 = bbox_.x;
      double x1 = bbox_.x + x_diff*0.5;
      double y0 = bbox_.y;
      double y1 = bbox_.y + y_diff*0.5;

      double w = x_diff*0.5;
      double h = y_diff*0.5;
      Children * new_children
          =
          new Children(
            new QuadTreeNode(
              Rectangle(x0,y0,w,h),
              qtree_->get_new_id(),
              qtree_),
            new QuadTreeNode(
              Rectangle(x0,y1,w,h),
              qtree_->get_new_id(),
              qtree_),
            new QuadTreeNode(
              Rectangle(x1,y0,w,h),
              qtree_->get_new_id(),
              qtree_),
            new QuadTreeNode(
              Rectangle(x1,y1,w,h),
              qtree_->get_new_id(),
              qtree_));
      new_children->insert(elem,bbox_);
      inserted = new_children->insert(new_elem,bbox_);
      children_ = new_children;
    }
  }
  else
  {
    inserted = children_->insert(new_elem,bbox_);
  }
  return inserted;


}

template<typename T>
void QuadTreeNode<T>::query(const Rectangle & win,
                            typename ALIGNED<QuadTreeElement<T> >::list
                            * content_list) const
{
  if (children_ == 0){
    if(is_empty_ == false){
      if (win.contains(cv::Point2d(elem.pos[0],elem.pos[1])))
      {
        content_list->push_back(elem);
      }
    }
  }
  else
  {
    if (intersects(children_->xy->bbox_,win))
    {
      children_->xy->query(win,content_list);
    }
    if (intersects(children_->xY->bbox_,win))
    {
      children_->xY->query(win,content_list);
    }
    if (intersects(children_->Xy->bbox_,win))
    {
      children_->Xy->query(win,content_list);
    }
    if (intersects(children_->XY->bbox_,win))
    {
      children_->XY->query(win,content_list);
    }
  }
}

template<typename T>
bool QuadTreeNode<T>::isWindowEmpty(const Rectangle & win) const
{
  if (children_ == 0)
  {
    if(is_empty_)
    {
      return true;
    }
    else
    {
      if (win.contains(cv::Point2d(elem.pos[0],elem.pos[1])))
      {
        return false;
      }
      return true;
    }
  }
  else
  {
    if (intersects(children_->xy->bbox_,win))
    {
      if (!children_->xy->isWindowEmpty(win))
        return false;
    }
    if (intersects(children_->xY->bbox_,win))
    {
      if (!children_->xY->isWindowEmpty(win))
        return false;
    }
    if (intersects(children_->Xy->bbox_,win))
    {
      if (!children_->Xy->isWindowEmpty(win))
        return false;
    }
    if (intersects(children_->XY->bbox_,win))
    {
      if (!children_->XY->isWindowEmpty(win))
        return false;
    }
  }
  return true;
}

template<typename T>
void QuadTreeNode<T>::traverse(list<QuadTreeElement<T> > & elem_list,
                               list<Rectangle> & quad_list) const
{
  quad_list.push_back(bbox_);
  assert(false);

}

}

#endif
