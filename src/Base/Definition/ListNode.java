package Base.Definition;

/**
 * @ClassName: ListNode
 * @Description:
 * @author: SeanLeong
 * @date: 2020/10/9 9:54
 */
public class ListNode {
    public int val;
    public ListNode next;

    public ListNode() {
    }

    public ListNode(int x) {
         val = x;
         next = null;
    }

    public ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}
