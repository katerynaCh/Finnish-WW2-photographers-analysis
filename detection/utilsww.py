import numpy as np

def non_max_suppression(boxes, scores, threshold):
        """Performs non-maximum suppression and returns indices of kept boxes.
        boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
        scores: 1-D array of box scores.
        threshold: Float. IoU threshold to use for filtering.
        """
        assert boxes.shape[0] > 0
        if boxes.dtype.kind != "f":
                boxes = boxes.astype(np.float32)
                # Compute box areas
        y1 = boxes[:, 0]
        x1 = boxes[:, 1]
        y2 = boxes[:, 2]
        x2 = boxes[:, 3]
        area = (y2 - y1) * (x2 - x1)

        # Get indicies of boxes sorted by scores (highest first)
        ixs = scores.argsort()[::-1]

        pick = []
        while len(ixs) > 0:
                # Pick top box and add its index to the list
                i = ixs[0]
                pick.append(i)
                # Compute IoU of the picked box with the rest
                iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
                # Identify boxes with IoU over the threshold. This
                # returns indices into ixs[1:], so add 1 to get
                # indices into ixs.
                remove_ixs = np.where(iou > threshold)[0] + 1
                # Remove indices of the picked and overlapped boxes.
                ixs = np.delete(ixs, remove_ixs)
                ixs = np.delete(ixs, 0)
        return np.array(pick, dtype=np.int32)   

def mean_boxes(boxes, scores, threshold):
        """Performs non-maximum suppression and returns indices of kept boxes.
        boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
        scores: 1-D array of box scores.
        threshold: Float. IoU threshold to use for filtering.
        """
        assert boxes.shape[0] > 0
        out_boxes = []
        if boxes.dtype.kind != "f":
                boxes = boxes.astype(np.float32)
                # Compute box areas
        y1 = boxes[:, 0]
        x1 = boxes[:, 1]
        y2 = boxes[:, 2]
        x2 = boxes[:, 3]
        area = (y2 - y1) * (x2 - x1)

        # Get indicies of boxes sorted by scores (highest first)
        ixs = scores.argsort()[::-1]

        pick = []
        while len(ixs) > 0:
                # Pick top box and add its index to the list
                i = ixs[0]
                #pick.append(i)
                # Compute IoU of the picked box with the rest
                iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
                # Identify boxes with IoU over the threshold. This
                # returns indices into ixs[1:], so add 1 to get
                # indices into ixs.
                same_ixs = np.where(iou > threshold)[0] + 1
                y1s = [boxes[i,0]]
                x1s = [boxes[i,1]]
                y2s = [boxes[i,2]]
                x2s = [boxes[i,3]]
                scoress = [float(scores[i])]
                for s in same_ixs:
                        y1s.append(boxes[ixs[s],0])
                        x1s.append(boxes[ixs[s],1])
                        y2s.append(boxes[ixs[s],2])
                        x2s.append(boxes[ixs[s],3])
                        scoress.append(float(scores[ixs[s]]))
                x1 = np.round(np.mean(np.asarray(x1s)))
                y1 = np.round(np.mean(np.asarray(y1s)))
                x2 = np.round(np.mean(np.asarray(x2s)))
                y2 = np.round(np.mean(np.asarray(y2s)))
                out_boxes.append([y1, x1, y2, x2, np.mean(np.asarray(scoress))])
                        
                # Remove indices of the picked and overlapped boxes.
                ixs = np.delete(ixs, same_ixs)
                ixs = np.delete(ixs, 0)
                
        return out_boxes                
        

def compute_iou(box, boxes, box_area, boxes_area):
        """Calculates IoU of the given box with the array of the given boxes.
        box: 1D vector [y1, x1, y2, x2]
        boxes: [boxes_count, (y1, x1, y2, x2)]
        box_area: float. the area of 'box'
        boxes_area: array of length boxes_count.

        Note: the areas are passed in rather than calculated here for
        efficiency. Calculate once in the caller to avoid duplicate work.
        """
        # Calculate intersection areas
        y1 = np.maximum(box[0], boxes[:, 0])
        y2 = np.minimum(box[2], boxes[:, 2])
        x1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[3], boxes[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = box_area + boxes_area[:] - intersection[:]
        iou = intersection / union
        return iou


