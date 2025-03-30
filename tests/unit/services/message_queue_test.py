import unittest
from parameterized import parameterized
from unittest.mock import patch, MagicMock, call
from services.message_queue import PriorityQueueManager, DeadLetterHandler

class TestMessageQueueService(unittest.TestCase):
    
    TEST_MESSAGES = [
        ({"id": 1, "priority": 0}, "low_priority"),
        ({"id": 2, "priority": 9}, "high_priority"),
        ({"id": 3, "priority": 5}, "medium_priority"),
        (None, "invalid_message")
    ]
    
    def setUp(self):
        self.dlq_handler = DeadLetterHandler(max_retries=3)
        self.queue = PriorityQueueManager(
            capacity=1000,
            dlq_handler=self.dlq_handler
        )
        
    @parameterized.expand(TEST_MESSAGES)
    def test_message_prioritization(self, message, case_name):
        if message:
            self.queue.enqueue(message)
            dequeued = self.queue.dequeue()
            self.assertEqual(dequeued["id"], message["id"])
        else:
            with self.assertRaises(ValueError):
                self.queue.enqueue(message)
                
    def test_capacity_overflow_handling(self):
        for i in range(1001):
            self.queue.enqueue({"id": i, "priority": i%10})
            
        self.assertEqual(self.dlq_handler.dlq_count(), 1)
        
    @patch('services.message_queue.DeadLetterHandler.retry_failed')
    def test_dead_letter_retry_mechanism(self, mock_retry):
        mock_retry.return_value = True
        failed_msg = MagicMock(attempts=0)
        
        for _ in range(3):
            self.dlq_handler.handle(failed_msg)
            
        self.assertEqual(failed_msg.attempts, 3)
        mock_retry.assert_called_once()

if __name__ == '__main__':
    unittest.TestLoader().loadTestsFromTestCase(TestMessageQueueService)
