package Base.Definition;

import com.sun.org.glassfish.gmbal.Description;
/**
 *
 @ClassName: TreeNode
 @Description:
 @author: SeanLeong
 @date: 2020/10/12 9:53
 *
 */
public class TreeNode {
    public int val;
    public TreeNode left;
    public TreeNode right;
    public TreeNode(int x) { val = x; }
    public TreeNode() {}
    public TreeNode(int val, TreeNode left, TreeNode right) {
      this.val = val;
      this.left = left;
      this.right = right;
    }
}
