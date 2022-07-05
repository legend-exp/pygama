import sys
import pika

def main():

    # test1()
    test2()


def test1():

    # cpars = pika.ConnectionParameters(host='10.66.193.71:15672')
    cpars = pika.ConnectionParameters(host='localhost')
    connection = pika.BlockingConnection(cpars)
    channel = connection.channel()

    qname = 'sensor_value.#'

    channel.queue_declare(queue=qname)

    def callback(ch, method, properties, body):
        print(" [x] Received %r" % body)

    channel.basic_consume(
        queue='sensor_value.mj60_baseline',
        on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


def test2():

    cpars = pika.ConnectionParameters(host='localhost')
    connection = pika.BlockingConnection()
    channel = connection.channel()

    exc_name = 'alerts'
    queue_name = "walter" # arbitrary.  the queue associated with THIS program

    # things going to the DB are always on the alerts queue and
    # always 'sensor_value.#'

    channel.exchange_declare(exchange='alerts', exchange_type='topic')

    channel.queue_declare(queue=queue_name, exclusive=True)

    channel.queue_bind(queue=queue_name, exchange='alerts', routing_key='sensor_value.#')

    print(' [*] Waiting for logs. To exit press CTRL+C')

    def callback(ch, method, properties, body):
        print(" [x] %r:%r" % (method.routing_key, body))
        # my_list.append(thing)

    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
    channel.start_consuming()


if __name__=="__main__":
    main()
