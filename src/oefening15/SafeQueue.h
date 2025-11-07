#include <shared_mutex>
#include <condition_variable>

/*
 * Als eerst gebruik gemaakt van een shared mutex aangezien deze code onder verschillende threads moet kunnen runnen.
 * Bij push en pop moet dit unique lock zijn, aangezien enkel 1 thread dat tegelijk mag doen.
 * Voor de reads zoals get en getSize mag een shared lock zijn. Dit mogen verschillende threads doen tegelijk.
 * Ik heb de busy waiting veranderd door een condition variable. 
 * In heb in de full_c en empty_c wel de code handmatig geschreven ipv gebruik te maken van getSize
 * Ik vond die getSize overbodig, aangezien de waits beide heel makkelijk zijn.
 * Ook zou dit nog best ingewikkeld zijn, aangezien in de wait al een lock is, dus dan zou ik een private getSize moeten maken die unsafe is
 * en een public getSize die wel een mutex heeft. Ik vind dit persoonlijk de beste oplossing.``
 * Hierna nog de notify one om duidelijk te maken dat een andere thread terug mag beginnen als het aan het wachten was. 
*/

class SafeQueue {
public:
    SafeQueue() : firstElement(0), lastElement(0) {}

    void push(float f) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        full_c.wait(lock, [this]() { 
            unsigned int next = (lastElement + 1) % queueSize;
            return next != firstElement;
        });

        queue[lastElement] = f;
        lastElement = (lastElement + 1) % queueSize;
        empty_c.notify_one();
    }

    float pop() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        empty_c.wait(lock, [this]() { 
            return firstElement != lastElement;
        });

        float value = queue[firstElement];
        firstElement = (firstElement + 1) % queueSize;
        full_c.notify_one();
        return value;
    }

    float get(unsigned int i) const {
        std::shared_lock lock(mutex_);
        return queue[(firstElement + i) % queueSize];
    }

    unsigned int getSize() const {
        std::shared_lock lock(mutex_);
        if (lastElement >= firstElement)
            return lastElement - firstElement;
        else
            return queueSize - (firstElement - lastElement);
    }

private:
    unsigned int firstElement, lastElement;
    static const unsigned int queueSize = 32;
    float queue[queueSize];

    mutable std::shared_mutex mutex_;
    std::condition_variable_any full_c, empty_c;
};
